use crate::error::ImageAnalysisError;
use crate::people::{PersonInfo, PersonObservation, RelationObservation};
use serde::Serialize;
use tokio_postgres::Client as PgClient;
use uuid::Uuid;

#[derive(Debug, Serialize)]
pub struct ImageAnalysisResult {
    pub description: String,
    pub asset_id: Uuid,
    #[serde(skip)]
    pub face_observations: Vec<PersonObservation>,
    #[serde(skip)]
    pub relation_observations: Vec<RelationObservation>,
}

/// Known Immich metadata for an asset (recognized faces with estimated age,
/// known relations between them, location) that can be used to enrich the
/// prompt sent to the vision model.
#[derive(Debug, Default)]
pub struct AssetContext {
    pub location: Option<String>,
    pub photo_date: Option<String>,
    pub photo_year: Option<i32>,
    pub persons: Vec<PersonInfo>,
    /// (sentence, certain) — certain is true for user-verified or logically
    /// inferred-from-verified facts, false for the co-occurrence heuristic's guesses.
    pub relations: Vec<(String, bool)>,
}

/// Turn a face bounding box into a coarse, human-readable position ("en haut à
/// gauche", "au centre", ...) based on which third of the image its center falls
/// into. Returns None if the reference image dimensions are unknown (0).
fn bounding_box_position(x1: i32, y1: i32, x2: i32, y2: i32, width: i32, height: i32) -> Option<String> {
    if width <= 0 || height <= 0 {
        return None;
    }
    let center_x = (x1 + x2) as f64 / 2.0;
    let center_y = (y1 + y2) as f64 / 2.0;
    let horizontal = match center_x / width as f64 {
        r if r < 1.0 / 3.0 => "à gauche",
        r if r < 2.0 / 3.0 => "au centre",
        _ => "à droite",
    };
    let vertical = match center_y / height as f64 {
        r if r < 1.0 / 3.0 => "en haut",
        r if r < 2.0 / 3.0 => "au milieu",
        _ => "en bas",
    };
    Some(format!("{vertical} {horizontal}"))
}

/// Fetch recognized faces (with estimated age at the time of this photo), known
/// relations between them, and location, to enrich the analysis prompt.
/// Best-effort: any database error is logged and yields an empty/partial context
/// rather than failing the whole image analysis.
pub async fn get_asset_context(client: &PgClient, asset_id: Uuid) -> AssetContext {
    let asset_id_str = asset_id.to_string();

    let location_query = "
        SELECT e.city, e.state, e.country,
               EXTRACT(YEAR FROM COALESCE(e.\"dateTimeOriginal\", a.\"fileCreatedAt\"))::int AS photo_year,
               TO_CHAR(COALESCE(e.\"dateTimeOriginal\", a.\"fileCreatedAt\"), 'DD/MM/YYYY') AS photo_date
        FROM asset a
        LEFT JOIN asset_exif e ON e.\"assetId\" = a.id
        WHERE a.id::text = $1
    ";
    let (location, photo_year, photo_date) = match client.query_opt(location_query, &[&asset_id_str]).await {
        Ok(Some(row)) => {
            let city: Option<String> = row.get(0);
            let state: Option<String> = row.get(1);
            let country: Option<String> = row.get(2);
            let photo_year: Option<i32> = row.get(3);
            let photo_date: Option<String> = row.get(4);
            let parts: Vec<String> = [city, state, country].into_iter().flatten().collect();
            let location = if parts.is_empty() { None } else { Some(parts.join(", ")) };
            (location, photo_year, photo_date)
        }
        Ok(None) => (None, None, None),
        Err(e) => {
            eprintln!(
                "{}",
                rust_i18n::t!("database.error_fetching_location", error = e.to_string())
            );
            (None, None, None)
        }
    };

    let faces_query = "
        SELECT DISTINCT ON (p.id) p.id::text, p.name,
               af.\"boundingBoxX1\", af.\"boundingBoxY1\", af.\"boundingBoxX2\", af.\"boundingBoxY2\",
               af.\"imageWidth\", af.\"imageHeight\"
        FROM asset_face af
        JOIN person p ON p.id = af.\"personId\"
        WHERE af.\"assetId\"::text = $1
          AND af.\"deletedAt\" IS NULL
          AND af.\"isVisible\" = true
          AND p.name != ''
          AND p.\"isHidden\" = false
        ORDER BY p.id, p.name
    ";
    let known_people: Vec<(String, String, Option<String>)> =
        match client.query(faces_query, &[&asset_id_str]).await {
            Ok(rows) => rows
                .iter()
                .map(|row| {
                    let id: String = row.get(0);
                    let name: String = row.get(1);
                    let x1: i32 = row.get(2);
                    let y1: i32 = row.get(3);
                    let x2: i32 = row.get(4);
                    let y2: i32 = row.get(5);
                    let width: i32 = row.get(6);
                    let height: i32 = row.get(7);
                    let position = bounding_box_position(x1, y1, x2, y2, width, height);
                    (id, name, position)
                })
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
    for (id_str, name, position) in &known_people {
        let Ok(id) = id_str.parse::<Uuid>() else {
            continue;
        };
        let profile_query = "
            SELECT birth_year_estimate, birth_year_spread, gender FROM analyze_person_profile WHERE person_id::text = $1
        ";
        let profile_row = client.query_opt(profile_query, &[id_str]).await.ok().flatten();
        let estimated_age = match (&profile_row, photo_year) {
            (Some(row), Some(year)) => {
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
        let gender = profile_row.as_ref().and_then(|row| {
            let gender: Option<String> = row.get(2);
            match gender.as_deref() {
                Some("male") => Some("garçon".to_string()),
                Some("female") => Some("fille".to_string()),
                _ => None,
            }
        });
        persons.push(PersonInfo {
            id,
            name: name.clone(),
            estimated_age,
            position: position.clone(),
            gender,
        });
    }

    let mut relations = if persons.len() >= 2 {
        fetch_relations(client, &persons).await
    } else {
        Vec::new()
    };
    if persons.len() >= 2 {
        relations.extend(fetch_visual_relation_hints(client, &persons).await);
    }

    AssetContext {
        location,
        photo_date,
        photo_year,
        persons,
        relations,
    }
}

/// A relation sentence paired with whether it's a user-confirmed fact (true) or
/// just the best guess of the co-occurrence/age-gap heuristic (false, "probable").
async fn fetch_relations(client: &PgClient, persons: &[PersonInfo]) -> Vec<(String, bool)> {
    let ids: Vec<String> = persons.iter().map(|p| p.id.to_string()).collect();
    let query = "
        SELECT person_id_a::text, person_id_b::text, relation_type, (verified OR inferred) AS certain
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
    let gender_of = |id: &str| -> Option<&str> {
        persons
            .iter()
            .find(|p| p.id.to_string() == id)
            .and_then(|p| p.gender.as_deref())
    };

    rows.iter()
        .filter_map(|row| {
            let id_a: String = row.get(0);
            let id_b: String = row.get(1);
            let relation_type: String = row.get(2);
            let certain: bool = row.get(3);
            let name_a = name_of(&id_a)?;
            let name_b = name_of(&id_b)?;
            let sentence = match relation_type.as_str() {
                "siblings" => format!("{} et {} sont probablement frère(s)/sœur(s).", name_a, name_b),
                "a_parent_of_b" => format!("{} est probablement un parent de {}.", name_a, name_b),
                "b_parent_of_a" => format!("{} est probablement un parent de {}.", name_b, name_a),
                "cousins" => format!("{} et {} sont cousins/cousines.", name_a, name_b),
                "friends" => format!("{} et {} sont ami(e)s.", name_a, name_b),
                "spouses" => format!("{} et {} sont en couple.", name_a, name_b),
                "a_aunt_uncle_of_b" => match gender_of(&id_a) {
                    Some("garçon") => format!("{} est l'oncle de {}.", name_a, name_b),
                    Some("fille") => format!("{} est la tante de {}.", name_a, name_b),
                    _ => format!("{} est oncle ou tante de {}.", name_a, name_b),
                },
                "b_aunt_uncle_of_a" => match gender_of(&id_b) {
                    Some("garçon") => format!("{} est l'oncle de {}.", name_b, name_a),
                    Some("fille") => format!("{} est la tante de {}.", name_b, name_a),
                    _ => format!("{} est oncle ou tante de {}.", name_b, name_a),
                },
                "sibling_in_law" => match gender_of(&id_a) {
                    Some("garçon") => format!("{} est le beau-frère de {}.", name_a, name_b),
                    Some("fille") => format!("{} est la belle-sœur de {}.", name_a, name_b),
                    _ => format!("{} et {} sont beau-frère/belle-sœur.", name_a, name_b),
                },
                "a_grandparent_of_b" => match gender_of(&id_a) {
                    Some("garçon") => format!("{} est le grand-père de {}.", name_a, name_b),
                    Some("fille") => format!("{} est la grand-mère de {}.", name_a, name_b),
                    _ => format!("{} est grand-parent de {}.", name_a, name_b),
                },
                "b_grandparent_of_a" => match gender_of(&id_b) {
                    Some("garçon") => format!("{} est le grand-père de {}.", name_b, name_a),
                    Some("fille") => format!("{} est la grand-mère de {}.", name_b, name_a),
                    _ => format!("{} est grand-parent de {}.", name_b, name_a),
                },
                "a_parent_in_law_of_b" => match gender_of(&id_a) {
                    Some("garçon") => format!("{} est le beau-père de {}.", name_a, name_b),
                    Some("fille") => format!("{} est la belle-mère de {}.", name_a, name_b),
                    _ => format!("{} est beau-père ou belle-mère de {}.", name_a, name_b),
                },
                "b_parent_in_law_of_a" => match gender_of(&id_b) {
                    Some("garçon") => format!("{} est le beau-père de {}.", name_b, name_a),
                    Some("fille") => format!("{} est la belle-mère de {}.", name_b, name_a),
                    _ => format!("{} est beau-père ou belle-mère de {}.", name_b, name_a),
                },
                "frequent_companion" => {
                    format!("{} et {} apparaissent très souvent ensemble.", name_a, name_b)
                }
                _ => return None,
            };
            Some((sentence, certain))
        })
        .collect()
}

/// Feed strong visually-observed relation hints (see `people::detect_visual_relation_hints`)
/// back into the prompt of *other* photos featuring the same pair, as "probable" —
/// never "confirmed". This is the read side of the loop: nothing is ever written
/// to `analyze_person_relation` from these observations (see that function's doc
/// comment for why an automatic write proved risky), so recomputing this at
/// prompt-build time, live from `analyze_relation_observation`, is the only place
/// this signal reaches the model at all.
async fn fetch_visual_relation_hints(client: &PgClient, persons: &[PersonInfo]) -> Vec<(String, bool)> {
    let ids: Vec<String> = persons.iter().map(|p| p.id.to_string()).collect();
    let query = "
        WITH per_pair_type AS (
            SELECT person_id_a, person_id_b, relation_type, COUNT(DISTINCT asset_id) AS agreement_count
            FROM analyze_relation_observation
            WHERE person_id_a::text = ANY($1) AND person_id_b::text = ANY($1)
            GROUP BY person_id_a, person_id_b, relation_type
        ),
        per_pair_total AS (
            SELECT person_id_a, person_id_b, COUNT(DISTINCT asset_id) AS photo_count
            FROM analyze_relation_observation
            WHERE person_id_a::text = ANY($1) AND person_id_b::text = ANY($1)
            GROUP BY person_id_a, person_id_b
        )
        SELECT t.person_id_a::text, t.person_id_b::text, t.relation_type
        FROM per_pair_type t
        JOIN per_pair_total tot ON tot.person_id_a = t.person_id_a AND tot.person_id_b = t.person_id_b
        WHERE tot.photo_count >= $2
          AND t.agreement_count::float8 / tot.photo_count >= $3
          AND NOT EXISTS (
              SELECT 1 FROM analyze_person_relation r
              WHERE r.person_id_a = t.person_id_a AND r.person_id_b = t.person_id_b AND (r.verified OR r.inferred)
          )
    ";
    let rows = match client
        .query(
            query,
            &[&ids, &crate::people::MIN_HINT_PHOTOS, &crate::people::MIN_HINT_AGREEMENT_RATIO],
        )
        .await
    {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!("Failed to fetch visual relation hints: {}", e);
            return Vec::new();
        }
    };

    let name_of = |id: &str| -> Option<&str> {
        persons.iter().find(|p| p.id.to_string() == id).map(|p| p.name.as_str())
    };

    rows.iter()
        .filter_map(|row| {
            let id_a: String = row.get(0);
            let id_b: String = row.get(1);
            let relation_type: String = row.get(2);
            let name_a = name_of(&id_a)?;
            let name_b = name_of(&id_b)?;
            let sentence = match relation_type.as_str() {
                "couple" => format!("{} et {} semblent former un couple (observé visuellement sur plusieurs photos).", name_a, name_b),
                "parent_enfant" => format!("{} et {} semblent avoir un lien parent-enfant (observé visuellement sur plusieurs photos).", name_a, name_b),
                "fratrie" => format!("{} et {} semblent être frère(s)/sœur(s) (observé visuellement sur plusieurs photos).", name_a, name_b),
                "oncle_tante" => format!("{} et {} semblent avoir un lien oncle/tante - neveu/nièce (observé visuellement sur plusieurs photos).", name_a, name_b),
                "amis" => format!("{} et {} semblent être ami(e)s (observé visuellement sur plusieurs photos).", name_a, name_b),
                _ => return None,
            };
            // Always "probable": this is the model's own visual guesswork accumulated
            // across photos, not a fact anyone confirmed.
            Some((sentence, false))
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
            .map(|p| {
                let age = match p.estimated_age {
                    Some((min, max)) if min == max => Some(format!("probablement {} ans", min)),
                    Some((min, max)) => Some(format!("probablement entre {} et {} ans", min, max)),
                    None => None,
                };
                let gender = p.gender.clone();
                let position = p.position.as_ref().map(|pos| format!("{} de l'image", pos));
                let details: Vec<String> = [gender, age, position].into_iter().flatten().collect();
                if details.is_empty() {
                    p.name.clone()
                } else {
                    format!("{} ({})", p.name, details.join(", "))
                }
            })
            .collect();
        context_lines.push(format!(
            "Personnes reconnues sur cette photo : {}.",
            people_desc.join(", ")
        ));
    }
    let confirmed_relations: Vec<&str> = context
        .relations
        .iter()
        .filter(|(_, certain)| *certain)
        .map(|(s, _)| s.as_str())
        .collect();
    let probable_relations: Vec<&str> = context
        .relations
        .iter()
        .filter(|(_, certain)| !*certain)
        .map(|(s, _)| s.as_str())
        .collect();
    if !confirmed_relations.is_empty() {
        context_lines.push(format!(
            "Liens confirmés (certains) entre ces personnes : {}",
            confirmed_relations.join(" ")
        ));
    }
    if !probable_relations.is_empty() {
        context_lines.push(format!(
            "Liens probables mais NON confirmés entre ces personnes (simples suppositions statistiques \
            à mentionner avec prudence si tu les utilises, ex. \"semblent être...\", \"peut-être...\") : {}",
            probable_relations.join(" ")
        ));
    }
    if let Some(date) = &context.photo_date {
        context_lines.push(format!("Date de la prise de vue : {}.", date));
    }
    if let Some(location) = &context.location {
        context_lines.push(format!("Lieu de la prise de vue : {}.", location));
    }
    if context_lines.is_empty() {
        return base_prompt.to_string();
    }
    let mut prompt = format!(
        "{}\n\nContexte connu (issu des métadonnées Immich) à utiliser pour enrichir la description, en mentionnant naturellement les noms, âges et lieu sans jamais dire qu'il s'agit de métadonnées ou de reconnaissance faciale. Les liens \"confirmés\" sont sûrs, traite-les comme des faits ; les liens \"probables\" sont de simples suppositions statistiques, formule-les avec prudence ou ignore-les si ça ne rend pas la description naturelle :\n{}",
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
    if context.persons.len() >= 2 {
        prompt.push_str(
            "\n\nSi tu observes un indice visuel clair (pas une simple supposition) sur la relation \
            entre deux des personnes nommées ci-dessus sur CETTE photo précise (se tiennent la main \
            comme un couple, embrassade typiquement parent-enfant, ressemblance familiale nette, \
            jeu entre enfants du même âge...), ajoute une ligne par paire concernée après les lignes \
            ###VISAGE (aucune ligne si tu n'as pas d'indice visuel net, ne devine pas) :\n\
            ###RELATION|Nom1|Nom2|type\n\
            où type vaut exactement l'un de : couple, parent_enfant, fratrie, oncle_tante, amis.",
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
