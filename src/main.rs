use clap::Parser;
use std::{path::Path, sync::Arc};
use tokio_postgres::{Client as PgClient, NoTls};

mod args;
mod config;
mod database;
mod error;
mod file_processing;
mod http_server;
mod llamacpp;
mod monitor;
mod ollama;
mod people;
mod progress;
mod utils;

use args::Args;
use config::MonitorConfig;
use file_processing::{get_immich_preview_files, handle_no_files, process_files_concurrently};
use monitor::monitor_folder;
use progress::SimpleProgress;
use utils::{determine_locale, get_system_locale, validate_args, validate_immich_directory};

rust_i18n::i18n!("locales", fallback = "en");

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger to enable debug logging
    env_logger::init();
    
    let system_locale = get_system_locale();
    let available_locales = rust_i18n::available_locales!();
    let args = Args::parse();
    let final_locale = determine_locale(&args.lang, &system_locale, &available_locales);
    rust_i18n::set_locale(&final_locale);
    println!(
        "{}",
        rust_i18n::t!("autodetect.locale_selected", locale = final_locale)
    );
    validate_args(&args)?;
    let immich_root = Path::new(&args.immich_root);
    validate_immich_directory(immich_root)?;
    let (pg_client, connection) = tokio_postgres::connect(&args.postgres_url, NoTls).await?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!(
                "{}",
                rust_i18n::t!("error.postgres_connection_error", error = e.to_string())
            );
        }
    });
    let pg_client_arc = Arc::new(pg_client);
    println!(
        "{}",
        rust_i18n::t!("main.postgres_connected", url = args.postgres_url)
    );
    if let Err(e) = database::check_database_connection(&pg_client_arc).await {
        eprintln!(
            "{}",
            rust_i18n::t!("error.database_connection_failed", error = e.to_string())
        );
        std::process::exit(1);
    }
    if let Err(e) = people::ensure_schema(&pg_client_arc).await {
        eprintln!(
            "{}",
            rust_i18n::t!("error.database_connection_failed", error = e.to_string())
        );
        std::process::exit(1);
    }
    if args.combined {
        run_combined_mode(immich_root, args.clone(), &pg_client_arc, &final_locale).await?;
    } else if args.monitor {
        run_monitor_mode(immich_root, &args, &pg_client_arc, &final_locale).await?;
    } else {
        run_batch_mode(immich_root, &args, &pg_client_arc, &final_locale).await?;
    }
    Ok(())
}

async fn run_combined_mode(
    immich_root: &Path,
    args: Args,
    pg_client: &Arc<PgClient>,
    locale: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", rust_i18n::t!("main.combined_mode_activated"));
    let batch_handle = {
        let immich_root = immich_root.to_path_buf();
        let args = args.clone();
        let pg_client = Arc::clone(pg_client);
        let locale = locale.to_string();
        tokio::spawn(async move {
            println!("{}", rust_i18n::t!("main.processing_existing_images"));
            if let Err(e) = run_batch_mode(&immich_root, &args, &pg_client, &locale).await {
                eprintln!(
                    "{}",
                    rust_i18n::t!("error.batch_mode_failed", error = e.to_string())
                );
            }
            println!("{}", rust_i18n::t!("main.batch_mode_completed"));
        })
    };
    let trigger_handle = spawn_trigger_server(immich_root, &args, pg_client);

    println!(
        "{}",
        rust_i18n::t!("main.monitor_mode_started_in_background")
    );
    run_monitor_mode(immich_root, &args, pg_client, locale).await?;
    let _ = batch_handle.await;
    if let Some(handle) = trigger_handle {
        handle.abort();
    }
    Ok(())
}

/// Starts the internal trigger HTTP server (see `http_server`) plus the task
/// that runs `run_relations_feedback_loop` on demand each time it receives a
/// /trigger request — so an external tool (e.g. the family-graph web UI, after
/// a human edits a relation) can ask for the consequences to be propagated right
/// away instead of waiting for this container's next restart. Returns None if
/// disabled (`--http-trigger-port 0`).
fn spawn_trigger_server(immich_root: &Path, args: &Args, pg_client: &Arc<PgClient>) -> Option<tokio::task::JoinHandle<()>> {
    if args.http_trigger_port == 0 {
        return None;
    }
    let immich_root = immich_root.to_path_buf();
    let args = args.clone();
    let pg_client = Arc::clone(pg_client);
    let port = args.http_trigger_port;

    Some(tokio::spawn(async move {
        let trigger_state = Arc::new(http_server::TriggerState::default());
        let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(1);

        let server_handle = tokio::spawn(http_server::serve(port, Arc::clone(&trigger_state), tx));

        let http_client = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(args.timeout))
            .build()
        {
            Ok(client) => client,
            Err(e) => {
                eprintln!("Failed to build HTTP client for trigger loop: {}", e);
                return;
            }
        };

        while rx.recv().await.is_some() {
            if !trigger_state.try_start() {
                continue;
            }
            println!("Trigger received: running relations feedback loop on demand.");
            run_relations_feedback_loop(&immich_root, &args, &pg_client, &http_client).await;
            trigger_state.finish();
        }
        let _ = server_handle.await;
    }))
}

async fn run_monitor_mode(
    immich_root: &Path,
    args: &Args,
    pg_client: &Arc<PgClient>,
    locale: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", rust_i18n::t!("main.monitor_mode_activated"));
    if args.ignore_existing {
        println!("{}", rust_i18n::t!("main.ignore_existing_enabled"));
    }
    let monitor_config = MonitorConfig {
        file_write_timeout: args.file_write_timeout,
        file_check_interval: args.file_check_interval,
        event_cooldown: args.event_cooldown,
        timeout: args.timeout,
        lang: locale.to_string(),
        ignore_existing: args.ignore_existing,
        hosts: args.hosts.clone(),
        interface: args.interface.clone(),
        api_key: args.api_key.clone(),
        unavailable_duration: args.unavailable_duration,
    };
    monitor_folder(
        immich_root,
        &args.model_name,
        Arc::clone(pg_client),
        &args.prompt,
        &monitor_config,
    )
    .await?;
    Ok(())
}

async fn run_batch_mode(
    immich_root: &Path,
    args: &Args,
    pg_client: &Arc<PgClient>,
    locale: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        rust_i18n::t!(
            "main.database_connected",
            path = "Immich PostgreSQL database"
        )
    );
    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(args.timeout))
        .build()?;

    if args.people_only {
        println!("People-only mode: skipping image description generation");
    } else {
        let preview_files = get_immich_preview_files(immich_root)?;
        handle_no_files(&preview_files, args.ignore_existing, immich_root)?;
        println!(
            "{}",
            rust_i18n::t!(
                "main.images_to_process",
                count = preview_files.len().to_string()
            )
        );
        println!(
            "{}",
            rust_i18n::t!("main.model_name", name = args.model_name)
        );
        println!(
            "{}",
            rust_i18n::t!(
                "main.max_concurrent",
                count = args.max_concurrent.to_string()
            )
        );
        println!(
            "{}",
            rust_i18n::t!("main.timeout", seconds = args.timeout.to_string())
        );
        if args.ignore_existing {
            println!("{}", rust_i18n::t!("main.ignore_existing_enabled"));
        }
        let progress = Arc::new(tokio::sync::Mutex::new(SimpleProgress::new(
            preview_files.len() as u64,
            &rust_i18n::t!("progress.processing_complete"),
        )));
        let results = process_files_concurrently(
            preview_files,
            &http_client,
            pg_client,
            args,
            locale,
            progress,
        )
        .await;
        file_processing::display_results(&results, args.max_concurrent > 1)?;
    }

    run_relations_feedback_loop(immich_root, args, pg_client, &http_client).await;

    Ok(())
}

/// Recompute knowledge -> reprocess the photos it affects -> those
/// freshly-regenerated descriptions can themselves refine/confirm/refute other
/// people's ages, relations or visual-relation hints -> recompute again. Capped
/// at MAX_FEEDBACK_LOOP_ITERATIONS so a genuinely large, slowly-settling family
/// graph can't turn into a runaway reprocessing loop; in practice it converges
/// (affected_persons becomes empty) well before the cap in almost every run,
/// since each pass only reacts to what the previous pass changed.
///
/// Called both at the end of a normal batch/combined run, and on-demand by the
/// HTTP trigger server (see `http_server`) when e.g. a relation is edited through
/// the companion web UI and needs its consequences propagated without waiting
/// for the container's next restart.
async fn run_relations_feedback_loop(immich_root: &Path, args: &Args, pg_client: &Arc<PgClient>, http_client: &reqwest::Client) {
    const MAX_FEEDBACK_LOOP_ITERATIONS: usize = 5;

    for iteration in 1..=MAX_FEEDBACK_LOOP_ITERATIONS {
        if iteration > 1 {
            println!("--- Feedback loop iteration {}/{} ---", iteration, MAX_FEEDBACK_LOOP_ITERATIONS);
        }

        let changed_persons = match people::recompute_profiles(pg_client).await {
            Ok(changed) => {
                println!(
                    "{}",
                    rust_i18n::t!("main.profiles_recomputed", count = changed.len().to_string())
                );
                changed
            }
            Err(e) => {
                eprintln!(
                    "{}",
                    rust_i18n::t!("error.profiles_recompute_failed", error = e.to_string())
                );
                Vec::new()
            }
        };
        let changed_relation_pairs = match people::recompute_relations(pg_client).await {
            Ok(changed) => {
                println!(
                    "{}",
                    rust_i18n::t!("main.relations_recomputed", count = changed.len().to_string())
                );
                changed
            }
            Err(e) => {
                eprintln!(
                    "{}",
                    rust_i18n::t!("error.relations_recompute_failed", error = e.to_string())
                );
                Vec::new()
            }
        };

        // Read-only: logs candidates for a human to review and, if confirmed, insert
        // as a verified fact. Never mutates the database, so it never feeds the
        // reprocessing cascade below.
        match people::detect_relation_contradictions(pg_client).await {
            Ok(contradictions) => println!("Relation contradictions detected (not auto-applied): {}", contradictions.len()),
            Err(e) => eprintln!("Failed to detect relation contradictions: {}", e),
        }

        // Also read-only: relations the vision model itself claims to visually
        // observe (see ###RELATION in the prompt), aggregated across photos.
        match people::detect_visual_relation_hints(pg_client).await {
            Ok(hints) => {
                println!("Visual relation hints detected (not auto-applied): {}", hints.len());
                for hint in &hints {
                    println!(
                        "  {} <-> {}: {} ({}/{} photos agree)",
                        hint.person_a, hint.person_b, hint.relation_type, hint.agreement_count, hint.photo_count
                    );
                }
            }
            Err(e) => eprintln!("Failed to detect visual relation hints: {}", e),
        }

        let inferred_relation_pairs = match people::infer_derived_relations(pg_client).await {
            Ok(touched) => {
                println!("Inferred relations touched: {}", touched.len());
                touched
            }
            Err(e) => {
                eprintln!("Failed to infer derived relations: {}", e);
                Vec::new()
            }
        };

        let mut affected_persons: Vec<uuid::Uuid> = changed_persons;
        for (a, b) in changed_relation_pairs.into_iter().chain(inferred_relation_pairs) {
            affected_persons.push(a);
            affected_persons.push(b);
        }
        affected_persons.sort();
        affected_persons.dedup();

        if affected_persons.is_empty() {
            if iteration > 1 {
                println!("Feedback loop converged after {} iteration(s): nothing left to refine.", iteration);
            }
            break;
        }

        let reprocessed_any =
            cascade_reprocess_affected_photos(http_client, pg_client, immich_root, &affected_persons, args).await;
        if !reprocessed_any {
            // Either the bulk-change circuit breaker tripped, or there were simply
            // no already-described photos to refresh — looping again would just
            // repeat the same recompute for no new effect.
            break;
        }
        if iteration == MAX_FEEDBACK_LOOP_ITERATIONS {
            println!(
                "Feedback loop hit the {}-iteration cap — stopping here even though more changes may remain.",
                MAX_FEEDBACK_LOOP_ITERATIONS
            );
        }
    }
}

/// Feedback loop: when a person's estimated age/relations changed materially,
/// regenerate the description of the already-described photos they appear in, so
/// the new knowledge gets woven in. Bounded on two sides to avoid a reprocessing
/// storm: a circuit breaker skips the cascade entirely when too many people
/// changed at once (a bulk backfill, not incremental refinement), and the number
/// of photos reprocessed per run is capped.
const MAX_CHANGED_PERSONS_FOR_CASCADE: usize = 20;
const MAX_ASSETS_TO_REPROCESS_PER_RUN: i64 = 200;

/// Returns true if at least one photo was actually reprocessed (used by the
/// feedback loop above to decide whether looping again could possibly help).
async fn cascade_reprocess_affected_photos(
    http_client: &reqwest::Client,
    pg_client: &Arc<PgClient>,
    immich_root: &Path,
    affected_persons: &[uuid::Uuid],
    args: &Args,
) -> bool {
    if affected_persons.is_empty() {
        return false;
    }
    if affected_persons.len() > MAX_CHANGED_PERSONS_FOR_CASCADE {
        println!(
            "{}",
            rust_i18n::t!(
                "main.cascade_skipped_bulk_change",
                count = affected_persons.len().to_string(),
                threshold = MAX_CHANGED_PERSONS_FOR_CASCADE.to_string()
            )
        );
        return false;
    }

    let asset_ids = match people::find_reprocessable_assets(
        pg_client,
        affected_persons,
        MAX_ASSETS_TO_REPROCESS_PER_RUN,
    )
    .await
    {
        Ok(ids) => ids,
        Err(e) => {
            eprintln!(
                "{}",
                rust_i18n::t!("error.cascade_lookup_failed", error = e.to_string())
            );
            return false;
        }
    };
    if asset_ids.is_empty() {
        return false;
    }
    println!(
        "{}",
        rust_i18n::t!("main.cascade_reprocessing_started", count = asset_ids.len().to_string())
    );
    if asset_ids.len() as i64 >= MAX_ASSETS_TO_REPROCESS_PER_RUN {
        println!(
            "{}",
            rust_i18n::t!(
                "main.cascade_reprocessing_capped",
                limit = MAX_ASSETS_TO_REPROCESS_PER_RUN.to_string()
            )
        );
    }

    let mut success = 0u32;
    let mut failed = 0u32;
    for asset_id in asset_ids {
        match file_processing::reprocess_asset(http_client, pg_client, immich_root, asset_id, args).await {
            Ok(_) => success += 1,
            Err(e) => {
                failed += 1;
                eprintln!(
                    "{}",
                    rust_i18n::t!(
                        "error.cascade_reprocess_failed",
                        asset_id = asset_id.to_string(),
                        error = e.to_string()
                    )
                );
            }
        }
    }
    println!(
        "{}",
        rust_i18n::t!(
            "main.cascade_reprocessing_complete",
            success = success.to_string(),
            failed = failed.to_string()
        )
    );
    success > 0
}
