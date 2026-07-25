use crate::error::ImageAnalysisError;
use crate::people::{PersonInfo, PersonObservation};
use serde::Serialize;
use tokio_postgres::Client as PgClient;
use uuid::Uuid;

#[derive(Debug, Serialize)]
pub struct ImageAnalysisResult {
    pub description: String,
    pub asset_id: Uuid,
    #[serde(skip)]
    pub face_observations: Vec<PersonObservation>,
}

/// Known Immich metadata for an asset (recognized faces with estimated age,
/// known relations between them, location) that can be used to enrich the
/// prompt sent to the vision model.
#[derive(Debug, Default)]
pub struct AssetContext {
    pub location: Option<String>,
    pub photo_year: Option<i32>,
    pub persons: Vec<PersonInfo>,
    pub relations: Vec<String>,
}

/// Fetch recognized faces (with estimated age at the time of this photo), known
/// relations between them, and location, to enrich the analysis prompt.
/// Best-effort: any database error is logged and yields an empty/partial context
/// rather than failing the whole image analysis.
pub async fn get_asset_context(client: &PgClient, asset_id: Uuid) -> AssetContext {
    let asset_id_str = asset_id.to_string();

    let location_query = "
        SELECT e.city, e.state, e.country,
               EXTRACT(YEAR FROM COALESCE(e.\"dateTimeOriginal\", a.\"fileCreatedAt\"))::int AS photo_year
        FROM asset a
        LEFT JOIN asset_exif e ON e.\"assetId\" = a.id
        WHERE a.id::text = $1
    ";
    let (location, photo_year) = match client.query_opt(location_query, &[&asset_id_str]).await {
        Ok(Some(row)) => {
            let city: Option<String> = row.get(0);
            let state: Option<String> = row.get(1);
            let country: Option<String> = row.get(2);
            let photo_year: Option<i32> = row.get(3);
            let parts: Vec<String> = [city, state, country].into_iter().flatten().collect();
            let location = if parts.is_empty() { None } else { Some(parts.join(", ")) };
            (location, photo_year)
        }
        Ok(None) => (None, None),
        Err(e) => {
            eprintln!(
                "{}",
                rust_i18n::t!("database.error_fetching_location", error = e.to_string())
            );
            (None, None)
        }
    };

    let faces_query = "
        SELECT DISTINCT p.id::text, p.name
        FROM asset_face af
        JOIN person p ON p.id = af.\"personId\"
        WHERE af.\"assetId\"::text = $1
          AND af.\"deletedAt\" IS NULL
          AND af.\"isVisible\" = true
          AND p.name != ''
          AND p.\"isHidden\" = false
        ORDER BY p.name
    ";
    let known_people: Vec<(String, String)> = match client.query(faces_query, &[&asset_id_str]).await {
        Ok(rows) => rows
            .iter()
            .map(|row| (row.get::<_, String>(0), row.get::<_, String>(1)))
            .collect(),
        Err(e) => {
            eprintln!(
                "{}",
                rust_i18n::t!("database.error_fetching_faces", error = e.to_string())
            );
            Vec::new()
        }
    };

    let mut persons = Vec::with_capacity(known_people.len());
    for (id_str, name) in &known_people {
        let Ok(id) = id_str.parse::<Uuid>() else {
            continue;
        };
        let profile_query = "
            SELECT birth_year_estimate, birth_year_spread FROM analyze_person_profile WHERE person_id::text = $1
        ";
        let estimated_age = match (client.query_opt(profile_query, &[id_str]).await, photo_year) {
            (Ok(Some(row)), Some(year)) => {
                let birth_year_estimate: Option<i32> = row.get(0);
                let birth_year_spread: i32 = row.get(1);
                birth_year_estimate.map(|birth_year| {
                    let age = year - birth_year;
                    let age_min = (age - birth_year_spread).max(0);
                    let age_max = (age + birth_year_spread).max(0);
                    (age_min, age_max)
                })
            }
            _ => None,
        };
        persons.push(PersonInfo {
            id,
            name: name.clone(),
            estimated_age,
        });
    }

    let relations = if persons.len() >= 2 {
        fetch_relations(client, &persons).await
    } else {
        Vec::new()
    };

    AssetContext {
        location,
        photo_year,
        persons,
        relations,
    }
}

async fn fetch_relations(client: &PgClient, persons: &[PersonInfo]) -> Vec<String> {
    let ids: Vec<String> = persons.iter().map(|p| p.id.to_string()).collect();
    let query = "
        SELECT person_id_a::text, person_id_b::text, relation_type
        FROM analyze_person_relation
        WHERE person_id_a::text = ANY($1) AND person_id_b::text = ANY($1)
          AND confidence >= 0.3
    ";
    let rows = match client.query(query, &[&ids]).await {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!(
                "{}",
                rust_i18n::t!("database.error_fetching_relations", error = e.to_string())
            );
            return Vec::new();
        }
    };

    let name_of = |id: &str| -> Option<&str> {
        persons
            .iter()
            .find(|p| p.id.to_string() == id)
            .map(|p| p.name.as_str())
    };

    rows.iter()
        .filter_map(|row| {
            let id_a: String = row.get(0);
            let id_b: String = row.get(1);
            let relation_type: String = row.get(2);
            let name_a = name_of(&id_a)?;
            let name_b = name_of(&id_b)?;
            let sentence = match relation_type.as_str() {
                "siblings" => format!("{} et {} sont probablement frère(s)/sœur(s).", name_a, name_b),
                "a_parent_of_b" => format!("{} est probablement un parent de {}.", name_a, name_b),
                "b_parent_of_a" => format!("{} est probablement un parent de {}.", name_b, name_a),
                "frequent_companion" => {
                    format!("{} et {} apparaissent très souvent ensemble.", name_a, name_b)
                }
                _ => return None,
            };
            Some(sentence)
        })
        .collect()
}

/// Append known faces (with estimated age), relations and location as context to
/// the base prompt so the model can naturally weave them into the description. Also
/// asks the model to report, for each named person, an observed age bracket and
/// gender in a parseable format, so the knowledge base keeps improving over time.
pub fn build_contextual_prompt(base_prompt: &str, context: &AssetContext) -> String {
    let mut context_lines = Vec::new();
    if !context.persons.is_empty() {
        let people_desc: Vec<String> = context
            .persons
            .iter()
            .map(|p| match p.estimated_age {
                Some((min, max)) if min == max => format!("{} (probablement {} ans)", p.name, min),
                Some((min, max)) => format!("{} (probablement entre {} et {} ans)", p.name, min, max),
                None => p.name.clone(),
            })
            .collect();
        context_lines.push(format!(
            "Personnes reconnues sur cette photo : {}.",
            people_desc.join(", ")
        ));
    }
    if !context.relations.is_empty() {
        context_lines.push(format!(
            "Liens connus entre ces personnes : {}",
            context.relations.join(" ")
        ));
    }
    if let Some(location) = &context.location {
        context_lines.push(format!("Lieu de la prise de vue : {}.", location));
    }
    if context_lines.is_empty() {
        return base_prompt.to_string();
    }
    let mut prompt = format!(
        "{}\n\nContexte connu (issu des métadonnées Immich) à utiliser pour enrichir la description, en mentionnant naturellement les noms, âges et lieu sans jamais dire qu'il s'agit de métadonnées ou de reconnaissance faciale :\n{}",
        base_prompt,
        context_lines.join("\n")
    );
    if !context.persons.is_empty() {
        prompt.push_str(
            "\n\nAprès la description, ajoute une ligne strictement au format suivant pour \
            chaque personne listée ci-dessus visible sur cette photo (une ligne par personne, \
            rien d'autre après) :\n\
            ###VISAGE|Nom|age_min|age_max|sexe\n\
            où age_min et age_max sont ta meilleure estimation en années de l'âge apparent sur \
            CETTE photo, et sexe vaut masculin, feminin ou inconnu.",
        );
    }
    prompt
}

/// Check if asset already has description in database
pub async fn asset_has_description(
    client: &PgClient,
    asset_id: Uuid,
) -> Result<bool, ImageAnalysisError> {
    let query = "
        SELECT EXISTS (
            SELECT 1 FROM asset_exif 
            WHERE \"assetId\"::text = $1 
            AND description IS NOT NULL 
            AND description != ''
        )
    ";
    let asset_id_str = asset_id.to_string();
    match client.query_one(query, &[&asset_id_str]).await {
        Ok(row) => Ok(row.get(0)),
        Err(e) => {
            eprintln!(
                "{}",
                rust_i18n::t!("database.error_checking_description", error = e.to_string())
            );
            Err(ImageAnalysisError::DatabaseError {
                error: e.to_string(),
            })
        }
    }
}

/// Update or create asset description in database
pub async fn update_or_create_asset_description(
    client: &PgClient,
    asset_id: Uuid,
    description: &str,
) -> Result<(), ImageAnalysisError> {
    let safe_description = description.replace("'", "''");
    let safe_asset_id = asset_id.to_string();
    println!(
        "{}",
        rust_i18n::t!("database.updating_asset", asset_id = asset_id.to_string())
    );
    let preview: String = description.chars().take(100).collect();
    println!(
        "{}",
        rust_i18n::t!(
            "database.description_length",
            length = description.len().to_string(),
            preview = preview
        )
    );

    let update_query = format!(
        r#"
        UPDATE asset_exif 
        SET description = E'{}', 
            "updatedAt" = NOW(),
            "updateId" = immich_uuid_v7()
        WHERE "assetId" = '{}'
        "#,
        safe_description, safe_asset_id
    );
    match client.execute(&update_query, &[]).await {
        Ok(rows_affected) => {
            if rows_affected > 0 {
                println!(
                    "{}",
                    rust_i18n::t!("database.update_success", asset_id = asset_id.to_string())
                );
                return Ok(());
            }
        }
        Err(e) => {
            eprintln!(
                "{}\n{}",
                rust_i18n::t!(
                    "database.update_error",
                    asset_id = asset_id.to_string(),
                    error = e.to_string()
                ),
                rust_i18n::t!("database.sql_query_details", query = update_query)
            );
            return Err(ImageAnalysisError::DatabaseError {
                error: e.to_string(),
            });
        }
    }

    let asset_exists_query = format!(
        r#"
        SELECT EXISTS (
            SELECT 1 FROM asset 
            WHERE id = '{}'
        )
        "#,
        safe_asset_id
    );
    let asset_exists = match client.query_one(&asset_exists_query, &[]).await {
        Ok(row) => row.get::<_, bool>(0),
        Err(e) => {
            eprintln!(
                "{}",
                rust_i18n::t!(
                    "database.asset_existence_check_error",
                    error = e.to_string()
                )
            );
            return Err(ImageAnalysisError::DatabaseError {
                error: e.to_string(),
            });
        }
    };
    if !asset_exists {
        eprintln!(
            "{}",
            rust_i18n::t!(
                "database.asset_not_in_table",
                asset_id = asset_id.to_string()
            )
        );
        return Err(ImageAnalysisError::DatabaseError {
            error: format!(
                "{}",
                rust_i18n::t!(
                    "database.asset_not_found_error",
                    asset_id = asset_id.to_string()
                )
            ),
        });
    }

    let insert_query = format!(
        r#"
        INSERT INTO asset_exif (
            "assetId", description, "updatedAt", "updateId"
        ) VALUES (
            '{}', E'{}', NOW(), immich_uuid_v7()
        )
        ON CONFLICT ("assetId") DO UPDATE 
        SET description = EXCLUDED.description,
            "updatedAt" = NOW(),
            "updateId" = immich_uuid_v7()
        "#,
        safe_asset_id, safe_description
    );

    match client.execute(&insert_query, &[]).await {
        Ok(_) => {
            println!(
                "{}",
                rust_i18n::t!("database.insert_success", asset_id = asset_id.to_string())
            );
            Ok(())
        }
        Err(e) => {
            eprintln!(
                "{}\n{}",
                rust_i18n::t!(
                    "database.insert_error",
                    asset_id = asset_id.to_string(),
                    error = e.to_string()
                ),
                rust_i18n::t!("database.sql_query_details", query = insert_query)
            );
            Err(ImageAnalysisError::DatabaseError {
                error: e.to_string(),
            })
        }
    }
}

pub async fn check_database_connection(client: &PgClient) -> Result<bool, ImageAnalysisError> {
    let timeout_duration = std::time::Duration::from_secs(5);
    match tokio::time::timeout(timeout_duration, client.query("SELECT 1", &[])).await {
        Ok(Ok(_)) => {
            println!("{}", rust_i18n::t!("database.connection_success"));
            Ok(true)
        }
        Ok(Err(e)) => {
            eprintln!(
                "{}",
                rust_i18n::t!("error.database_query_failed", error = e.to_string())
            );
            Err(ImageAnalysisError::DatabaseError {
                error: format!(
                    "{}",
                    rust_i18n::t!("database.query_failed_error", error = e.to_string())
                ),
            })
        }
        Err(_) => {
            eprintln!("{}", rust_i18n::t!("error.database_timeout"));
            Err(ImageAnalysisError::DatabaseError {
                error: format!("{}", rust_i18n::t!("database.timeout_error")),
            })
        }
    }
}
