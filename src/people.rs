use crate::error::ImageAnalysisError;
use log::{debug, warn};
use tokio_postgres::Client as PgClient;
use uuid::Uuid;

/// A person known to be present in a photo, with an age estimate for that
/// specific photo derived from their accumulated profile (if any).
#[derive(Debug, Clone)]
pub struct PersonInfo {
    pub id: Uuid,
    pub name: String,
    pub estimated_age: Option<(i32, i32)>,
}

/// A single raw age/gender observation for a person extracted from a model response.
#[derive(Debug, Clone)]
pub struct PersonObservation {
    pub person_id: Uuid,
    pub age_min: i32,
    pub age_max: i32,
    pub gender: Option<String>,
}

const OBSERVATION_MARKER: &str = "###VISAGE|";

/// Split a model response into the clean description (for asset_exif.description)
/// and the structured per-person age/gender observations, matched by name against
/// the persons known to be in this photo.
pub fn parse_response(raw: &str, persons: &[PersonInfo]) -> (String, Vec<PersonObservation>) {
    let mut description_lines = Vec::new();
    let mut observations = Vec::new();

    for line in raw.lines() {
        let trimmed = line.trim();
        let Some(payload) = trimmed.strip_prefix(OBSERVATION_MARKER) else {
            description_lines.push(line);
            continue;
        };
        let parts: Vec<&str> = payload.split('|').map(str::trim).collect();
        if parts.len() != 4 {
            warn!("Malformed VISAGE observation line: {}", trimmed);
            continue;
        }
        let (name, age_min, age_max, gender) = (parts[0], parts[1], parts[2], parts[3]);
        let Some(person) = persons
            .iter()
            .find(|p| p.name.eq_ignore_ascii_case(name))
        else {
            warn!("VISAGE observation for unknown person '{}', skipping", name);
            continue;
        };
        let (Ok(age_min), Ok(age_max)) = (age_min.parse::<i32>(), age_max.parse::<i32>()) else {
            warn!("Non-numeric age in VISAGE observation: {}", trimmed);
            continue;
        };
        let gender_lower = gender.to_lowercase();
        let gender = if gender_lower.contains("fem") {
            Some("female".to_string())
        } else if gender_lower.contains("masc") {
            Some("male".to_string())
        } else {
            None
        };
        observations.push(PersonObservation {
            person_id: person.id,
            age_min: age_min.min(age_max),
            age_max: age_min.max(age_max),
            gender,
        });
    }

    (description_lines.join("\n").trim().to_string(), observations)
}

/// Create the knowledge-base tables used to accumulate face observations and
/// derive person profiles/relations over time. Safe to call on every startup.
pub async fn ensure_schema(client: &PgClient) -> Result<(), ImageAnalysisError> {
    let statements = [
        r#"
        CREATE TABLE IF NOT EXISTS analyze_face_observation (
            asset_id UUID NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
            person_id UUID NOT NULL REFERENCES person(id) ON DELETE CASCADE,
            photo_year INTEGER NOT NULL,
            age_min INTEGER NOT NULL,
            age_max INTEGER NOT NULL,
            gender VARCHAR,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (asset_id, person_id)
        )
        "#,
        r#"
        CREATE TABLE IF NOT EXISTS analyze_person_profile (
            person_id UUID PRIMARY KEY REFERENCES person(id) ON DELETE CASCADE,
            birth_year_estimate INTEGER,
            birth_year_spread INTEGER NOT NULL DEFAULT 0,
            gender VARCHAR,
            observation_count INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        "#,
        r#"
        CREATE TABLE IF NOT EXISTS analyze_person_relation (
            person_id_a UUID NOT NULL REFERENCES person(id) ON DELETE CASCADE,
            person_id_b UUID NOT NULL REFERENCES person(id) ON DELETE CASCADE,
            relation_type VARCHAR NOT NULL,
            confidence REAL NOT NULL,
            co_occurrence_count INTEGER NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (person_id_a, person_id_b)
        )
        "#,
    ];
    for statement in statements {
        client
            .batch_execute(statement)
            .await
            .map_err(|e| ImageAnalysisError::DatabaseError {
                error: e.to_string(),
            })?;
    }
    Ok(())
}

/// Persist the raw per-photo observations. Best-effort: logged and swallowed on error
/// so a knowledge-base hiccup never fails the (already successful) image analysis.
pub async fn record_observations(
    client: &PgClient,
    asset_id: Uuid,
    photo_year: Option<i32>,
    observations: &[PersonObservation],
) {
    let Some(photo_year) = photo_year else {
        return;
    };
    for obs in observations {
        let query = "
            INSERT INTO analyze_face_observation
                (asset_id, person_id, photo_year, age_min, age_max, gender)
            VALUES ($1::text::uuid, $2::text::uuid, $3, $4, $5, $6)
            ON CONFLICT (asset_id, person_id) DO UPDATE
            SET photo_year = EXCLUDED.photo_year,
                age_min = EXCLUDED.age_min,
                age_max = EXCLUDED.age_max,
                gender = EXCLUDED.gender
        ";
        let asset_id_str = asset_id.to_string();
        let person_id_str = obs.person_id.to_string();
        if let Err(e) = client
            .execute(
                query,
                &[
                    &asset_id_str,
                    &person_id_str,
                    &photo_year,
                    &obs.age_min,
                    &obs.age_max,
                    &obs.gender,
                ],
            )
            .await
        {
            warn!(
                "Failed to record face observation for person {}: {}",
                obs.person_id, e
            );
        }
    }
}

/// A birth-year estimate changing by at least this many years (or appearing for the
/// first time) is considered material enough to warrant re-describing that person's
/// existing photos.
const MATERIAL_BIRTH_YEAR_CHANGE: i32 = 2;

/// Recompute every person's estimated birth year from all accumulated observations.
/// Per-photo age guesses from the vision model are noisy (a single mislabeled or
/// oddly-cropped photo can be off by a decade), so instead of intersecting every
/// observation's possible-birth-year range (fragile: one outlier makes the whole
/// range empty/contradictory), we take the *median* birth-year implied by each
/// observation's midpoint age. The median is robust to outliers and tightens
/// naturally as more photos accumulate. Returns the ids of persons whose estimate
/// changed materially (new estimate, or shifted by MATERIAL_BIRTH_YEAR_CHANGE+ years).
pub async fn recompute_profiles(client: &PgClient) -> Result<Vec<Uuid>, ImageAnalysisError> {
    let before_rows = client
        .query(
            "SELECT person_id::text, birth_year_estimate FROM analyze_person_profile",
            &[],
        )
        .await
        .map_err(|e| ImageAnalysisError::DatabaseError {
            error: e.to_string(),
        })?;
    let before: std::collections::HashMap<String, Option<i32>> = before_rows
        .into_iter()
        .map(|row| (row.get::<_, String>(0), row.get::<_, Option<i32>>(1)))
        .collect();

    let query = "
        WITH point_estimates AS (
            SELECT
                person_id,
                photo_year - (age_min + age_max) / 2.0 AS birth_year_point,
                gender
            FROM analyze_face_observation
        )
        INSERT INTO analyze_person_profile
            (person_id, birth_year_estimate, birth_year_spread, gender, observation_count, updated_at)
        SELECT
            person_id,
            ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY birth_year_point))::int AS birth_year_estimate,
            ROUND((
                percentile_cont(0.75) WITHIN GROUP (ORDER BY birth_year_point) -
                percentile_cont(0.25) WITHIN GROUP (ORDER BY birth_year_point)
            ) / 2.0)::int AS birth_year_spread,
            mode() WITHIN GROUP (ORDER BY gender) FILTER (WHERE gender IS NOT NULL) AS gender,
            COUNT(*) AS observation_count,
            now()
        FROM point_estimates
        GROUP BY person_id
        ON CONFLICT (person_id) DO UPDATE
        SET birth_year_estimate = EXCLUDED.birth_year_estimate,
            birth_year_spread = EXCLUDED.birth_year_spread,
            gender = EXCLUDED.gender,
            observation_count = EXCLUDED.observation_count,
            updated_at = now()
    ";
    client
        .execute(query, &[])
        .await
        .map_err(|e| ImageAnalysisError::DatabaseError {
            error: e.to_string(),
        })?;

    let after_rows = client
        .query(
            "SELECT person_id::text, birth_year_estimate FROM analyze_person_profile",
            &[],
        )
        .await
        .map_err(|e| ImageAnalysisError::DatabaseError {
            error: e.to_string(),
        })?;

    let mut changed = Vec::new();
    for row in after_rows {
        let id_str: String = row.get(0);
        let new_estimate: Option<i32> = row.get(1);
        let old_estimate = before.get(&id_str).copied().flatten();
        let materially_changed = match (old_estimate, new_estimate) {
            (None, Some(_)) => true,
            (Some(old), Some(new)) => (old - new).abs() >= MATERIAL_BIRTH_YEAR_CHANGE,
            _ => false,
        };
        if materially_changed {
            if let Ok(id) = id_str.parse::<Uuid>() {
                changed.push(id);
            }
        }
    }
    debug!("{} person profiles changed materially", changed.len());
    Ok(changed)
}

async fn mid_birth_year(client: &PgClient, person_id: &str) -> Option<i32> {
    let query = "SELECT birth_year_estimate FROM analyze_person_profile WHERE person_id::text = $1";
    let row = client.query_opt(query, &[&person_id]).await.ok().flatten()?;
    row.get(0)
}

const MIN_CO_OCCURRENCE: i64 = 5;

/// Infer probable relations (siblings, parent/child, frequent companions) from how
/// often pairs of named people co-occur in photos, combined with their estimated
/// ages. Pure statistics/heuristics, no LLM call. Returns the (person_a, person_b)
/// pairs whose relation is new or changed type since last run.
pub async fn recompute_relations(client: &PgClient) -> Result<Vec<(Uuid, Uuid)>, ImageAnalysisError> {
    let before_rows = client
        .query(
            "SELECT person_id_a::text, person_id_b::text, relation_type FROM analyze_person_relation",
            &[],
        )
        .await
        .map_err(|e| ImageAnalysisError::DatabaseError {
            error: e.to_string(),
        })?;
    let before: std::collections::HashMap<(String, String), String> = before_rows
        .into_iter()
        .map(|row| {
            (
                (row.get::<_, String>(0), row.get::<_, String>(1)),
                row.get::<_, String>(2),
            )
        })
        .collect();

    let co_occurrence_query = "
        SELECT
            LEAST(af1.\"personId\", af2.\"personId\")::text AS person_a,
            GREATEST(af1.\"personId\", af2.\"personId\")::text AS person_b,
            COUNT(DISTINCT af1.\"assetId\") AS co_occurrence_count
        FROM asset_face af1
        JOIN asset_face af2
            ON af1.\"assetId\" = af2.\"assetId\" AND af1.\"personId\" < af2.\"personId\"
        JOIN person p1 ON p1.id = af1.\"personId\" AND p1.name != '' AND p1.\"isHidden\" = false
        JOIN person p2 ON p2.id = af2.\"personId\" AND p2.name != '' AND p2.\"isHidden\" = false
        WHERE af1.\"deletedAt\" IS NULL AND af1.\"isVisible\" = true
          AND af2.\"deletedAt\" IS NULL AND af2.\"isVisible\" = true
        GROUP BY person_a, person_b
        HAVING COUNT(DISTINCT af1.\"assetId\") >= $1
    ";
    let rows = client
        .query(co_occurrence_query, &[&MIN_CO_OCCURRENCE])
        .await
        .map_err(|e| ImageAnalysisError::DatabaseError {
            error: e.to_string(),
        })?;

    let mut changed = Vec::new();

    for row in rows {
        let person_a: String = row.get(0);
        let person_b: String = row.get(1);
        let co_occurrence_count: i64 = row.get(2);

        let birth_year_a = mid_birth_year(client, &person_a).await;
        let birth_year_b = mid_birth_year(client, &person_b).await;

        let (relation_type, confidence) = match (birth_year_a, birth_year_b) {
            (Some(ya), Some(yb)) => {
                let gap = (ya - yb).abs();
                if gap <= 6 {
                    ("siblings", (co_occurrence_count as f32 / 20.0).min(1.0))
                } else if gap >= 15 {
                    if ya < yb {
                        ("a_parent_of_b", (co_occurrence_count as f32 / 20.0).min(1.0))
                    } else {
                        ("b_parent_of_a", (co_occurrence_count as f32 / 20.0).min(1.0))
                    }
                } else {
                    ("frequent_companion", (co_occurrence_count as f32 / 30.0).min(1.0))
                }
            }
            _ => ("frequent_companion", (co_occurrence_count as f32 / 30.0).min(1.0)),
        };

        let upsert = "
            INSERT INTO analyze_person_relation
                (person_id_a, person_id_b, relation_type, confidence, co_occurrence_count, updated_at)
            VALUES ($1::text::uuid, $2::text::uuid, $3, $4, $5, now())
            ON CONFLICT (person_id_a, person_id_b) DO UPDATE
            SET relation_type = EXCLUDED.relation_type,
                confidence = EXCLUDED.confidence,
                co_occurrence_count = EXCLUDED.co_occurrence_count,
                updated_at = now()
        ";
        match client
            .execute(
                upsert,
                &[&person_a, &person_b, &relation_type, &confidence, &(co_occurrence_count as i32)],
            )
            .await
        {
            Ok(_) => {
                let previous = before.get(&(person_a.clone(), person_b.clone()));
                let is_new_or_changed = match previous {
                    Some(prev_type) => prev_type != relation_type,
                    None => true,
                };
                if is_new_or_changed {
                    if let (Ok(id_a), Ok(id_b)) = (person_a.parse::<Uuid>(), person_b.parse::<Uuid>()) {
                        changed.push((id_a, id_b));
                    }
                }
            }
            Err(e) => warn!(
                "Failed to upsert relation between {} and {}: {}",
                person_a, person_b, e
            ),
        }
    }
    debug!("{} person relations changed", changed.len());
    Ok(changed)
}

/// Find already-described photos featuring any of the given persons, so their
/// description can be regenerated with the newly learned age/relations context.
/// Capped at `limit` assets; callers should log if the true count exceeds it so
/// truncation is never silent.
pub async fn find_reprocessable_assets(
    client: &PgClient,
    person_ids: &[Uuid],
    limit: i64,
) -> Result<Vec<Uuid>, ImageAnalysisError> {
    if person_ids.is_empty() {
        return Ok(Vec::new());
    }
    let ids: Vec<String> = person_ids.iter().map(|id| id.to_string()).collect();
    let query = "
        SELECT DISTINCT af.\"assetId\"::text
        FROM asset_face af
        JOIN asset_exif e ON e.\"assetId\" = af.\"assetId\"
        WHERE af.\"personId\"::text = ANY($1)
          AND af.\"deletedAt\" IS NULL
          AND af.\"isVisible\" = true
          AND e.description IS NOT NULL
          AND e.description != ''
        LIMIT $2
    ";
    let rows = client
        .query(query, &[&ids, &limit])
        .await
        .map_err(|e| ImageAnalysisError::DatabaseError {
            error: e.to_string(),
        })?;
    Ok(rows
        .into_iter()
        .filter_map(|row| row.get::<_, String>(0).parse::<Uuid>().ok())
        .collect())
}
