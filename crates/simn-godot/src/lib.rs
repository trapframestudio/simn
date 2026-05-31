use godot::prelude::*;
use std::sync::Once;

mod los;
mod network;
mod physics_setup;
mod regional;
mod sim;
mod terrain;
mod util;

struct SimnExtension;

#[gdextension]
unsafe impl ExtensionLibrary for SimnExtension {
    fn on_stage_init(stage: InitStage) {
        if stage == InitStage::Scene {
            init_tracing();
        }
    }
}

static TRACING_INIT: Once = Once::new();

/// Install a tracing subscriber that routes to the Godot console
/// (`godot_print!`), filtered by `RUST_LOG` (default
/// `npc.behavior=info`). Idempotent; gated so re-init from hot-reload
/// doesn't double-install. Without this, all `tracing::info!` calls
/// in the sim are silent no-ops.
fn init_tracing() {
    TRACING_INIT.call_once(|| {
        use tracing_subscriber::{fmt, prelude::*, EnvFilter};
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("npc.behavior=info"));
        let layer = fmt::layer()
            .with_writer(GodotWriter)
            .with_target(false)
            .with_level(false)
            .without_time();
        let _ = tracing_subscriber::registry()
            .with(filter)
            .with(layer)
            .try_init();
    });
}

/// `std::io::Write` shim that forwards each line to `godot_print!`.
/// Godot's logger flushes asynchronously, so this doesn't stall the
/// main thread even when the sim is chatty.
struct GodotWriter;

impl std::io::Write for GodotWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let s = std::str::from_utf8(buf).unwrap_or("<non-utf8>");
        // Strip trailing newline — godot_print! adds its own.
        let s = s.strip_suffix('\n').unwrap_or(s);
        if !s.is_empty() {
            godot_print!("{s}");
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for GodotWriter {
    type Writer = GodotWriter;
    fn make_writer(&'a self) -> Self::Writer {
        GodotWriter
    }
}
