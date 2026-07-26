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
    pub position: Option<String>,
    pub gender: Option<String>,
}

/// A single raw age/gender observation for a person extracted from a model response.
#[derive(Debug, Clone)]
pub struct PersonObservation {
    pub person_id: Uuid,
    pub age_min: i32,
    pub age_max: i32,
    pub gender: Option<String>,
}

/// A relation the vision model claims to visually observe between two named
/// people on one specific photo (hand-holding, a parent-child embrace, visible
/// family resemblance...) — distinct from, and a lot noisier than, the
/// co-occurrence heuristic: a single photo's worth of evidence, from a model
/// that can be wrong or overconfident. Accumulated across many photos and only
/// ever surfaced as a suggestion (see `detect_visual_relation_hints`), never
/// auto-applied.
#[derive(Debug, Clone)]
pub struct RelationObservation {
    pub person_a: Uuid,
    pub person_b: Uuid,
    pub relation_type: String,
}

const OBSERVATION_MARKER: &str = "###VISAGE|";
const RELATION_OBSERVATION_MARKER: &str = "###RELATION|";
/// Canonical vocabulary the prompt asks the model to use for ###RELATION lines.
/// Anything else (including the model's own "incertain"/unsure marker) is dropped.
const VALID_OBSERVED_RELATION_TYPES: &[&str] = &["couple", "parent_enfant", "fratrie", "oncle_tante", "amis"];

/// Split a model response into the clean description (for asset_exif.description),
/// the structured per-person age/gender observations, and any observed relations
/// between two named people, matched by name against the persons known to be in
/// this photo.
pub fn parse_response(raw: &str, persons: &[PersonInfo]) -> (String, Vec<PersonObservation>, Vec<RelationObservation>) {
    let mut description_lines = Vec::new();
    let mut observations = Vec::new();
    let mut relation_observations = Vec::new();

    let find_person = |name: &str| persons.iter().find(|p| p.name.eq_ignore_ascii_case(name));

    for line in raw.lines() {
        let trimmed = line.trim();
        if let Some(payload) = trimmed.strip_prefix(RELATION_OBSERVATION_MARKER) {
            let parts: Vec<&str> = payload.split('|').map(str::trim).collect();
            if parts.len() != 3 {
                warn!("Malformed RELATION observation line: {}", trimmed);
                continue;
            }
            let (name_a, name_b, relation_type) = (parts[0], parts[1], parts[2]);
            let relation_type = relation_type.to_lowercase();
            if !VALID_OBSERVED_RELATION_TYPES.contains(&relation_type.as_str()) {
                // Includes the model reporting "incertain" (no clear visual evidence) —
                // silently skipped, that's the expected way to opt out of a pair.
                continue;
            }
            let (Some(person_a), Some(person_b)) = (find_person(name_a), find_person(name_b)) else {
                warn!("RELATION observation for unknown person(s) '{}'/'{}', skipping", name_a, name_b);
                continue;
            };
            if person_a.id == person_b.id {
                continue;
            }
            relation_observations.push(RelationObservation {
                person_a: person_a.id,
                person_b: person_b.id,
                relation_type,
            });
            continue;
        }
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
        let Some(person) = find_person(name) else {
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

    (description_lines.join("\n").trim().to_string(), observations, relation_observations)
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
        // Manually confirmed relations (e.g. entered by the user) must survive the
        // next automatic recompute, which only infers from age-gap/co-occurrence
        // heuristics and would otherwise silently overwrite them.
        "ALTER TABLE analyze_person_relation ADD COLUMN IF NOT EXISTS verified BOOLEAN NOT NULL DEFAULT false",
        // Relations produced by the transitive inference engine (infer_derived_relations):
        // logically derived from verified facts, not from photo statistics. Protected
        // from the co-occurrence heuristic the same way verified rows are, but (unlike
        // verified rows) freely recomputed by the inference engine itself on every run.
        "ALTER TABLE analyze_person_relation ADD COLUMN IF NOT EXISTS inferred BOOLEAN NOT NULL DEFAULT false",
        r#"
        CREATE TABLE IF NOT EXISTS analyze_relation_observation (
            asset_id UUID NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
            person_id_a UUID NOT NULL REFERENCES person(id) ON DELETE CASCADE,
            person_id_b UUID NOT NULL REFERENCES person(id) ON DELETE CASCADE,
            relation_type VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (asset_id, person_id_a, person_id_b)
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

/// Persist the raw per-photo visually-observed relations. Best-effort, same as
/// `record_observations`: never fails the (already successful) image analysis.
pub async fn record_relation_observations(client: &PgClient, asset_id: Uuid, observations: &[RelationObservation]) {
    for obs in observations {
        // Canonical (person_id_a < person_id_b) ordering, consistent regardless of
        // which order the model happened to name the two people in.
        let (id_a, id_b) = if obs.person_a < obs.person_b {
            (obs.person_a, obs.person_b)
        } else {
            (obs.person_b, obs.person_a)
        };
        let query = "
            INSERT INTO analyze_relation_observation (asset_id, person_id_a, person_id_b, relation_type)
            VALUES ($1::text::uuid, $2::text::uuid, $3::text::uuid, $4)
            ON CONFLICT (asset_id, person_id_a, person_id_b) DO UPDATE
            SET relation_type = EXCLUDED.relation_type, created_at = now()
        ";
        if let Err(e) = client
            .execute(query, &[&asset_id.to_string(), &id_a.to_string(), &id_b.to_string(), &obs.relation_type])
            .await
        {
            warn!("Failed to record relation observation for {}<->{}: {}", id_a, id_b, e);
        }
    }
}

/// A pair of people, and the visually-observed relation type the model has
/// converged on across multiple distinct photos — a suggestion for a human to
/// review and, if it seems right, insert as a verified fact. Never auto-applied
/// (see module docs on `detect_relation_contradictions` for why cross-referencing
/// noisy per-photo signals into an automatic write is risky).
#[derive(Debug)]
pub struct VisualRelationHint {
    pub person_a: Uuid,
    pub person_b: Uuid,
    pub relation_type: String,
    pub photo_count: i64,
    pub agreement_count: i64,
}

/// Minimum number of distinct photos where the model reported *some* relation
/// for a pair before it's worth surfacing at all — a single photo's guess is too
/// thin to act on. Also used by `database::fetch_visual_relation_hints` to decide
/// what's solid enough to feed back into the prompt as a "probable" relation.
pub(crate) const MIN_HINT_PHOTOS: i64 = 3;
/// Minimum fraction of those photos that must agree on the same relation_type.
pub(crate) const MIN_HINT_AGREEMENT_RATIO: f64 = 0.7;

/// Aggregate `analyze_relation_observation` into candidate suggestions: pairs
/// with enough photos, and where the model was reasonably consistent about what
/// it saw, are surfaced as `VisualRelationHint`s. Read-only — nothing is written
/// to `analyze_person_relation`.
pub async fn detect_visual_relation_hints(client: &PgClient) -> Result<Vec<VisualRelationHint>, ImageAnalysisError> {
    let query = "
        WITH per_pair_type AS (
            SELECT person_id_a, person_id_b, relation_type, COUNT(DISTINCT asset_id) AS agreement_count
            FROM analyze_relation_observation
            GROUP BY person_id_a, person_id_b, relation_type
        ),
        per_pair_total AS (
            SELECT person_id_a, person_id_b, COUNT(DISTINCT asset_id) AS photo_count
            FROM analyze_relation_observation
            GROUP BY person_id_a, person_id_b
        )
        SELECT t.person_id_a::text, t.person_id_b::text, t.relation_type, tot.photo_count, t.agreement_count
        FROM per_pair_type t
        JOIN per_pair_total tot ON tot.person_id_a = t.person_id_a AND tot.person_id_b = t.person_id_b
        WHERE tot.photo_count >= $1
          AND t.agreement_count::float8 / tot.photo_count >= $2
          AND NOT EXISTS (
              SELECT 1 FROM analyze_person_relation r
              WHERE r.person_id_a = t.person_id_a AND r.person_id_b = t.person_id_b AND (r.verified OR r.inferred)
          )
        ORDER BY tot.photo_count DESC
    ";
    let rows = client
        .query(query, &[&MIN_HINT_PHOTOS, &MIN_HINT_AGREEMENT_RATIO])
        .await
        .map_err(|e| ImageAnalysisError::DatabaseError {
            error: e.to_string(),
        })?;

    let mut hints = Vec::new();
    for row in rows {
        let id_a: String = row.get(0);
        let id_b: String = row.get(1);
        let relation_type: String = row.get(2);
        let photo_count: i64 = row.get(3);
        let agreement_count: i64 = row.get(4);
        let (Ok(person_a), Ok(person_b)) = (id_a.parse::<Uuid>(), id_b.parse::<Uuid>()) else {
            continue;
        };
        hints.push(VisualRelationHint {
            person_a,
            person_b,
            relation_type,
            photo_count,
            agreement_count,
        });
    }
    debug!("{} visual relation hints detected", hints.len());
    Ok(hints)
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

        // Never touch a relation the user has manually confirmed (verified = true)
        // or one produced by the transitive inference engine (inferred = true): the
        // automatic heuristic below only looks at age gap and co-occurrence count,
        // which can't tell siblings from cousins or spot friendships/in-laws.
        let upsert = "
            INSERT INTO analyze_person_relation
                (person_id_a, person_id_b, relation_type, confidence, co_occurrence_count, updated_at)
            VALUES ($1::text::uuid, $2::text::uuid, $3, $4, $5, now())
            ON CONFLICT (person_id_a, person_id_b) DO UPDATE
            SET relation_type = EXCLUDED.relation_type,
                confidence = EXCLUDED.confidence,
                co_occurrence_count = EXCLUDED.co_occurrence_count,
                updated_at = now()
            WHERE analyze_person_relation.verified = false
              AND analyze_person_relation.inferred = false
        ";
        match client
            .execute(
                upsert,
                &[&person_a, &person_b, &relation_type, &confidence, &(co_occurrence_count as i32)],
            )
            .await
        {
            Ok(rows_affected) => {
                if rows_affected == 0 {
                    continue;
                }
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

/// A person pair whose heuristic "siblings"/"cousins" guess contradicts another
/// heuristic guess that they share a child (incest isn't modeled, so sharing a
/// child implies a couple, not siblings/cousins). Read-only: this is a signal for
/// a human (or an admin tool) to review, not something safe to auto-apply — see
/// `detect_relation_contradictions` for why.
#[derive(Debug)]
pub struct RelationContradiction {
    pub person_a: Uuid,
    pub person_b: Uuid,
    pub previous_relation_type: String,
    pub shared_child: Uuid,
}

/// Surface (but do NOT rewrite) cases where the co-occurrence/age-gap heuristic's
/// "siblings"/"cousins" guess for a pair contradicts its separate "parent of"
/// guesses (both pointing at the same child). This is deliberately read-only:
/// an earlier version auto-corrected these to "spouses", but in practice even
/// high-confidence heuristic edges are noisy enough (an uncle/aunt/grandparent
/// frequently in family photos can get mistakenly flagged "parent of") that
/// cross-referencing two independently-noisy heuristics compounded errors
/// instead of catching real contradictions — it produced impossible results like
/// one person "married" to four different people. Treat the output as candidates
/// for a human to confirm (then insert as a verified fact), not ground truth.
pub async fn detect_relation_contradictions(client: &PgClient) -> Result<Vec<RelationContradiction>, ImageAnalysisError> {
    const MIN_CONTRADICTION_CONFIDENCE: f32 = 0.8;

    let candidate_rows = client
        .query(
            "SELECT person_id_a::text, person_id_b::text, relation_type FROM analyze_person_relation
             WHERE verified = false AND inferred = false AND relation_type IN ('siblings', 'cousins')
               AND confidence >= $1",
            &[&MIN_CONTRADICTION_CONFIDENCE],
        )
        .await
        .map_err(|e| ImageAnalysisError::DatabaseError {
            error: e.to_string(),
        })?;

    let parent_rows = client
        .query(
            "SELECT person_id_a::text, person_id_b::text, relation_type FROM analyze_person_relation
             WHERE relation_type IN ('a_parent_of_b', 'b_parent_of_a')
               AND (verified OR inferred OR confidence >= $1)",
            &[&MIN_CONTRADICTION_CONFIDENCE],
        )
        .await
        .map_err(|e| ImageAnalysisError::DatabaseError {
            error: e.to_string(),
        })?;

    let mut parents_of: std::collections::HashMap<Uuid, std::collections::HashSet<Uuid>> = std::collections::HashMap::new();
    for row in &parent_rows {
        let id_a: String = row.get(0);
        let id_b: String = row.get(1);
        let relation_type: String = row.get(2);
        let (Ok(a), Ok(b)) = (id_a.parse::<Uuid>(), id_b.parse::<Uuid>()) else {
            continue;
        };
        let (parent, child) = if relation_type == "a_parent_of_b" { (a, b) } else { (b, a) };
        parents_of.entry(child).or_default().insert(parent);
    }

    let mut contradictions = Vec::new();
    for row in &candidate_rows {
        let id_a: String = row.get(0);
        let id_b: String = row.get(1);
        let previous_relation_type: String = row.get(2);
        let (Ok(a), Ok(b)) = (id_a.parse::<Uuid>(), id_b.parse::<Uuid>()) else {
            continue;
        };
        let Some(shared_child) = parents_of
            .iter()
            .find(|(_, parents)| parents.contains(&a) && parents.contains(&b))
            .map(|(child, _)| *child)
        else {
            continue;
        };
        warn!(
            "Contradiction detected ({} <-> {}, currently '{}'): both heuristically parent of {} \
            — likely a couple, review and insert as a verified fact if confirmed",
            a, b, previous_relation_type, shared_child
        );
        contradictions.push(RelationContradiction {
            person_a: a,
            person_b: b,
            previous_relation_type,
            shared_child,
        });
    }
    debug!("{} relation contradictions detected", contradictions.len());
    Ok(contradictions)
}

/// Maximum number of rule-application passes for `infer_derived_relations`. Real
/// family graphs converge in 2-3 passes; this is just a defensive cap against an
/// unexpected cycle in the data.
const MAX_INFERENCE_PASSES: usize = 6;

/// Derive relations that are logically implied by the *verified* (user-confirmed)
/// relations already in the database, using a small fixed-point rule engine — not
/// photo statistics. Verified facts are the only accepted input on purpose: the
/// heuristic co-occurrence/age-gap guesses in `recompute_relations` are noisy, and
/// letting those feed the inference engine would silently propagate their errors
/// (e.g. a wrongly-guessed "siblings" turning into a chain of wrong cousins).
///
/// Rules applied to a fixed point:
///   - shared child                         => the two parents are spouses
///   - parent's sibling                     => aunt/uncle (blood)
///   - existing aunt/uncle's spouse         => aunt/uncle (by marriage)
///   - two parents who are siblings         => their children are cousins
///   - spouse's sibling                     => sibling-in-law
///   - existing sibling-in-law's spouse     => sibling-in-law (chained in-law)
///   - parent's parent                      => grandparent
///   - spouse's parent                      => parent-in-law
///
/// Every run fully replaces the previously inferred set (rows with inferred = true
/// and verified = false) with a fresh recomputation, so removing/correcting a
/// verified relation naturally retracts whatever it used to imply. Returns the ids
/// of persons touched by any newly-inserted or changed inferred relation, so callers
/// can feed them into the reprocessing cascade like any other changed relation.
pub async fn infer_derived_relations(client: &PgClient) -> Result<Vec<(Uuid, Uuid)>, ImageAnalysisError> {
    let rows = client
        .query(
            "SELECT person_id_a::text, person_id_b::text, relation_type FROM analyze_person_relation WHERE verified = true",
            &[],
        )
        .await
        .map_err(|e| ImageAnalysisError::DatabaseError {
            error: e.to_string(),
        })?;

    let mut parent_of: std::collections::HashSet<(Uuid, Uuid)> = std::collections::HashSet::new(); // (parent, child)
    let mut sibling: std::collections::HashSet<(Uuid, Uuid)> = std::collections::HashSet::new();
    let mut spouse: std::collections::HashSet<(Uuid, Uuid)> = std::collections::HashSet::new();
    let mut cousin: std::collections::HashSet<(Uuid, Uuid)> = std::collections::HashSet::new();
    let mut aunt_uncle: std::collections::HashSet<(Uuid, Uuid)> = std::collections::HashSet::new(); // (aunt/uncle, nephew/niece)
    let mut sibling_in_law: std::collections::HashSet<(Uuid, Uuid)> = std::collections::HashSet::new();
    let mut grandparent: std::collections::HashSet<(Uuid, Uuid)> = std::collections::HashSet::new(); // (grandparent, grandchild)
    let mut parent_in_law: std::collections::HashSet<(Uuid, Uuid)> = std::collections::HashSet::new(); // (parent-in-law, child-in-law)

    let add_symmetric = |set: &mut std::collections::HashSet<(Uuid, Uuid)>, a: Uuid, b: Uuid| -> bool {
        let mut changed = set.insert((a, b));
        changed |= set.insert((b, a));
        changed
    };

    for row in &rows {
        let id_a: String = row.get(0);
        let id_b: String = row.get(1);
        let relation_type: String = row.get(2);
        let (Ok(a), Ok(b)) = (id_a.parse::<Uuid>(), id_b.parse::<Uuid>()) else {
            continue;
        };
        match relation_type.as_str() {
            "siblings" => {
                add_symmetric(&mut sibling, a, b);
            }
            "a_parent_of_b" => {
                parent_of.insert((a, b));
            }
            "b_parent_of_a" => {
                parent_of.insert((b, a));
            }
            "spouses" => {
                add_symmetric(&mut spouse, a, b);
            }
            "cousins" => {
                add_symmetric(&mut cousin, a, b);
            }
            _ => {}
        }
    }

    // parents_of[child] -> parents; rebuilt once, used by the shared-child (spouse)
    // rule below. NOTE: keyed by child, not by parent — the rule pairs up two
    // *parents* of the same child, not two children of the same parent (that
    // would make siblings "spouses").
    let mut parents_of_child: std::collections::HashMap<Uuid, Vec<Uuid>> = std::collections::HashMap::new();
    let mut children_of_parent: std::collections::HashMap<Uuid, Vec<Uuid>> = std::collections::HashMap::new();
    for (parent, child) in &parent_of {
        parents_of_child.entry(*child).or_default().push(*parent);
        children_of_parent.entry(*parent).or_default().push(*child);
    }

    for _pass in 0..MAX_INFERENCE_PASSES {
        let mut changed = false;

        // Rule: two people who share a child are spouses.
        for parents in parents_of_child.values() {
            for i in 0..parents.len() {
                for j in (i + 1)..parents.len() {
                    if parents[i] != parents[j] && add_symmetric(&mut spouse, parents[i], parents[j]) {
                        changed = true;
                    }
                }
            }
        }

        // Rule: two different children of the same parent are siblings (the
        // converse of the spouse rule above).
        for children in children_of_parent.values() {
            for i in 0..children.len() {
                for j in (i + 1)..children.len() {
                    if children[i] != children[j] && add_symmetric(&mut sibling, children[i], children[j]) {
                        changed = true;
                    }
                }
            }
        }

        // Rule: a parent's sibling is a blood aunt/uncle.
        for (parent, child) in parent_of.clone() {
            for (s_a, s_b) in sibling.clone() {
                if s_a == parent && s_b != parent {
                    changed |= aunt_uncle.insert((s_b, child));
                }
            }
        }

        // Rule: an aunt/uncle's spouse is also an aunt/uncle (by marriage).
        for (auncle, child) in aunt_uncle.clone() {
            for (sp_a, sp_b) in spouse.clone() {
                if sp_a == auncle && sp_b != auncle {
                    changed |= aunt_uncle.insert((sp_b, child));
                }
            }
        }

        // Rule: children of siblings are cousins.
        for (parent_a, child_a) in parent_of.clone() {
            for (parent_b, child_b) in parent_of.clone() {
                if parent_a != parent_b && child_a != child_b && sibling.contains(&(parent_a, parent_b)) {
                    changed |= add_symmetric(&mut cousin, child_a, child_b);
                }
            }
        }

        // Rule: a parent's parent is a grandparent.
        for (parent, child) in parent_of.clone() {
            for (grandparent_id, parent2) in parent_of.clone() {
                if parent2 == parent {
                    changed |= grandparent.insert((grandparent_id, child));
                }
            }
        }

        // Rule: a spouse's parent is a parent-in-law.
        for (sp_a, sp_b) in spouse.clone() {
            for (parent, child) in parent_of.clone() {
                if child == sp_b {
                    changed |= parent_in_law.insert((parent, sp_a));
                }
            }
        }

        // Rule: a spouse's sibling is a sibling-in-law.
        for (sp_a, sp_b) in spouse.clone() {
            for (s_a, s_b) in sibling.clone() {
                if s_a == sp_b && s_b != sp_b {
                    changed |= add_symmetric(&mut sibling_in_law, sp_a, s_b);
                }
            }
        }

        // Rule: a sibling-in-law's spouse is also a sibling-in-law (chained, e.g.
        // "my wife's brother's wife").
        for (il_a, il_b) in sibling_in_law.clone() {
            for (sp_a, sp_b) in spouse.clone() {
                if sp_a == il_b && sp_b != il_b && sp_b != il_a {
                    changed |= add_symmetric(&mut sibling_in_law, il_a, sp_b);
                }
            }
        }

        if !changed {
            break;
        }
    }

    // Never claim someone is their own relative, and don't emit an "inferred"
    // relation for a pair that already has a stricter, more specific relation
    // (e.g. two people who are both siblings and, through some quirk of the data,
    // would also match the cousins rule).
    let has_closer_relation = |a: Uuid, b: Uuid| sibling.contains(&(a, b)) || parent_of.contains(&(a, b)) || parent_of.contains(&(b, a));

    fn push_symmetric(
        set: &std::collections::HashSet<(Uuid, Uuid)>,
        rel: &'static str,
        derived: &mut Vec<(Uuid, Uuid, &'static str)>,
    ) {
        let mut seen_pairs: std::collections::HashSet<(Uuid, Uuid)> = std::collections::HashSet::new();
        for &(a, b) in set {
            if a == b {
                continue;
            }
            let key = if a < b { (a, b) } else { (b, a) };
            if !seen_pairs.insert(key) {
                continue;
            }
            derived.push((key.0, key.1, rel));
        }
    }
    // Blood siblings/parent-child always wins: a looser category (cousins,
    // sibling-in-law, even spouses) must never overwrite it in `derived` — the
    // chained sibling-in-law rule in particular can loop back through a spouse
    // link and "reclassify" two actual blood siblings as in-laws of each other
    // (e.g. two brothers who are each other's wife's brother-in-law by the
    // chain, even though they're plainly brothers already).
    let drop_if_closer = |set: &std::collections::HashSet<(Uuid, Uuid)>| -> std::collections::HashSet<(Uuid, Uuid)> {
        set.iter().copied().filter(|&(a, b)| !has_closer_relation(a, b)).collect()
    };
    let spouse = drop_if_closer(&spouse);
    let cousin = drop_if_closer(&cousin);
    let sibling_in_law = drop_if_closer(&sibling_in_law);

    let mut derived: Vec<(Uuid, Uuid, &'static str)> = Vec::new();
    push_symmetric(&sibling, "siblings", &mut derived);
    push_symmetric(&spouse, "spouses", &mut derived);
    push_symmetric(&cousin, "cousins", &mut derived);
    push_symmetric(&sibling_in_law, "sibling_in_law", &mut derived);

    for &(auncle, child) in &aunt_uncle {
        if auncle == child || has_closer_relation(auncle, child) {
            continue;
        }
        let (id_a, id_b, relation_type) = if auncle < child {
            (auncle, child, "a_aunt_uncle_of_b")
        } else {
            (child, auncle, "b_aunt_uncle_of_a")
        };
        derived.push((id_a, id_b, relation_type));
    }

    for &(gp, gc) in &grandparent {
        if gp == gc || has_closer_relation(gp, gc) {
            continue;
        }
        let (id_a, id_b, relation_type) = if gp < gc {
            (gp, gc, "a_grandparent_of_b")
        } else {
            (gc, gp, "b_grandparent_of_a")
        };
        derived.push((id_a, id_b, relation_type));
    }

    for &(il, child_in_law) in &parent_in_law {
        if il == child_in_law || has_closer_relation(il, child_in_law) {
            continue;
        }
        let (id_a, id_b, relation_type) = if il < child_in_law {
            (il, child_in_law, "a_parent_in_law_of_b")
        } else {
            (child_in_law, il, "b_parent_in_law_of_a")
        };
        derived.push((id_a, id_b, relation_type));
    }

    // Snapshot what was inferred *before* this pass, so "touched" below reflects an
    // actual before/after diff — not just "every row this pass happened to write",
    // which the DELETE+reinsert approach would otherwise report on every single run
    // even when nothing changed (that inflated affected_persons on every pass,
    // triggering a full reprocessing cascade even for a no-op run).
    let before_rows = client
        .query(
            "SELECT person_id_a::text, person_id_b::text, relation_type FROM analyze_person_relation WHERE inferred = true",
            &[],
        )
        .await
        .map_err(|e| ImageAnalysisError::DatabaseError {
            error: e.to_string(),
        })?;
    let before: std::collections::HashMap<(Uuid, Uuid), String> = before_rows
        .into_iter()
        .filter_map(|row| {
            let id_a: String = row.get(0);
            let id_b: String = row.get(1);
            let relation_type: String = row.get(2);
            Some(((id_a.parse().ok()?, id_b.parse().ok()?), relation_type))
        })
        .collect();

    // Replace the previously inferred set wholesale: any relation this pass no
    // longer derives (e.g. because a verified fact was corrected) is dropped, and
    // a verified row for the same pair is never touched.
    client
        .execute(
            "DELETE FROM analyze_person_relation WHERE inferred = true AND verified = false",
            &[],
        )
        .await
        .map_err(|e| ImageAnalysisError::DatabaseError {
            error: e.to_string(),
        })?;

    let mut touched = Vec::new();
    for (id_a, id_b, relation_type) in derived {
        let upsert = "
            INSERT INTO analyze_person_relation
                (person_id_a, person_id_b, relation_type, confidence, co_occurrence_count, updated_at, inferred)
            VALUES ($1::text::uuid, $2::text::uuid, $3, 0.95, 0, now(), true)
            ON CONFLICT (person_id_a, person_id_b) DO UPDATE
            SET relation_type = EXCLUDED.relation_type,
                confidence = EXCLUDED.confidence,
                inferred = true,
                updated_at = now()
            WHERE analyze_person_relation.verified = false
        ";
        match client
            .execute(upsert, &[&id_a.to_string(), &id_b.to_string(), &relation_type])
            .await
        {
            Ok(rows_affected) => {
                if rows_affected == 0 {
                    continue;
                }
                let is_new_or_changed = match before.get(&(id_a, id_b)) {
                    Some(prev_type) => prev_type != relation_type,
                    None => true,
                };
                if is_new_or_changed {
                    touched.push((id_a, id_b));
                }
            }
            Err(e) => warn!("Failed to upsert inferred relation between {} and {}: {}", id_a, id_b, e),
        }
    }
    debug!("{} inferred relations touched", touched.len());
    Ok(touched)
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
