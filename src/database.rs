use crate::error::ImageAnalysisError;
use serde::Serialize;
use tokio_postgres::Client as PgClient;
use uuid::Uuid;

#[derive(Debug, Serialize)]
pub struct ImageAnalysisResult {
    pub description: String,
    pub asset_id: Uuid,
}

/// Known Immich metadata for an asset (recognized faces, location) that can
/// be used to enrich the prompt sent to the vision model.
#[derive(Debug, Default)]
pub struct AssetContext {
    pub location: Option<String>,
    pub person_names: Vec<String>,
}

/// Fetch recognized face names and location for an asset, to enrich the analysis prompt.
/// Best-effort: any database error is logged and yields an empty context rather than
/// failing the whole image analysis.
pub async fn get_asset_context(client: &PgClient, asset_id: Uuid) -> AssetContext {
    let asset_id_str = asset_id.to_string();

    let location_query = "
        SELECT city, state, country
        FROM asset_exif
        WHERE \"assetId\"::text = $1
    ";
    let location = match client.query_opt(location_query, &[&asset_id_str]).await {
        Ok(Some(row)) => {
            let city: Option<String> = row.get(0);
            let state: Option<String> = row.get(1);
            let country: Option<String> = row.get(2);
            let parts: Vec<String> = [city, state, country].into_iter().flatten().collect();
            if parts.is_empty() { None } else { Some(parts.join(", ")) }
        }
        Ok(None) => None,
        Err(e) => {
            eprintln!(
                "{}",
                rust_i18n::t!("database.error_fetching_location", error = e.to_string())
            );
            None
        }
    };

    let faces_query = "
        SELECT DISTINCT p.name
        FROM asset_face af
        JOIN person p ON p.id = af.\"personId\"
        WHERE af.\"assetId\"::text = $1
          AND af.\"deletedAt\" IS NULL
          AND af.\"isVisible\" = true
          AND p.name != ''
          AND p.\"isHidden\" = false
        ORDER BY p.name
    ";
    let person_names = match client.query(faces_query, &[&asset_id_str]).await {
        Ok(rows) => rows.iter().map(|row| row.get::<_, String>(0)).collect(),
        Err(e) => {
            eprintln!(
                "{}",
                rust_i18n::t!("database.error_fetching_faces", error = e.to_string())
            );
            Vec::new()
        }
    };

    AssetContext {
        location,
        person_names,
    }
}

/// Append known faces/location as context to the base prompt so the model can
/// naturally weave them into the description.
pub fn build_contextual_prompt(base_prompt: &str, context: &AssetContext) -> String {
    let mut context_lines = Vec::new();
    if !context.person_names.is_empty() {
        context_lines.push(format!(
            "Personnes reconnues sur cette photo : {}.",
            context.person_names.join(", ")
        ));
    }
    if let Some(location) = &context.location {
        context_lines.push(format!("Lieu de la prise de vue : {}.", location));
    }
    if context_lines.is_empty() {
        return base_prompt.to_string();
    }
    format!(
        "{}\n\nContexte connu (issu des métadonnées Immich) à utiliser pour enrichir la description, en mentionnant naturellement les noms et le lieu sans jamais dire qu'il s'agit de métadonnées ou de reconnaissance faciale :\n{}",
        base_prompt,
        context_lines.join("\n")
    )
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
