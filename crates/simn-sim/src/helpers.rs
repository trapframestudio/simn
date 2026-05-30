//! Stray utility helpers that don't belong to a specific type.
//!
//! Historically these drifted into `components.rs` or `resources.rs`
//! even though they're pure free functions — not components, not
//! resources. Centralizing them here keeps those files focused on
//! type definitions.
//!
//! Everything here is re-exported from `lib.rs` under the same names
//! it had before the move, so no public API change.

use crate::components::BaseKind;
use crate::resources::Weather;

/// Stable string tag for a [`BaseKind`]. Used by the Godot bridge to
/// name bases in `bases_in_region`, and by the debug overlay.
pub fn base_kind_to_str(k: BaseKind) -> &'static str {
    match k {
        BaseKind::Checkpoint => "checkpoint",
        BaseKind::Outpost => "outpost",
        BaseKind::Safehouse => "safehouse",
        BaseKind::Headquarters => "headquarters",
        BaseKind::ResearchPost => "research_post",
        BaseKind::CampSite => "campsite",
    }
}

/// Human-readable moon phase name at the current fraction. Buckets
/// the continuous phase into the eight canonical names. Used by the
/// debug overlay and by future almanac UI.
pub fn moon_phase_name(phase: f32) -> &'static str {
    let p = phase.rem_euclid(1.0);
    match (p * 8.0).floor() as i32 {
        0 => "new",
        1 => "waxing crescent",
        2 => "first quarter",
        3 => "waxing gibbous",
        4 => "full",
        5 => "waning gibbous",
        6 => "last quarter",
        _ => "waning crescent",
    }
}

/// Stable snake_case tag for a [`Weather`] variant. Inverse of
/// [`weather_from_str`].
pub fn weather_to_str(w: Weather) -> &'static str {
    match w {
        Weather::Clear => "clear",
        Weather::PartlyCloudy => "partly_cloudy",
        Weather::Overcast => "overcast",
        Weather::MarineLayer => "marine_layer",
        Weather::Fog => "fog",
        Weather::Drizzle => "drizzle",
        Weather::LightRain => "light_rain",
        Weather::HeavyRain => "heavy_rain",
        Weather::Windstorm => "windstorm",
        Weather::Thunderstorm => "thunderstorm",
        Weather::SmokeHaze => "smoke_haze",
    }
}

/// Parse a weather string back into [`Weather`]. Case-insensitive,
/// accepts both `"light_rain"` and `"LightRain"` forms.
pub fn weather_from_str(s: &str) -> Option<Weather> {
    match s.to_ascii_lowercase().replace(' ', "_").as_str() {
        "clear" => Some(Weather::Clear),
        "partly_cloudy" | "partlycloudy" => Some(Weather::PartlyCloudy),
        "overcast" => Some(Weather::Overcast),
        "marine_layer" | "marinelayer" => Some(Weather::MarineLayer),
        "fog" => Some(Weather::Fog),
        "drizzle" => Some(Weather::Drizzle),
        "light_rain" | "lightrain" => Some(Weather::LightRain),
        "heavy_rain" | "heavyrain" => Some(Weather::HeavyRain),
        "windstorm" => Some(Weather::Windstorm),
        "thunderstorm" => Some(Weather::Thunderstorm),
        "smoke_haze" | "smokehaze" => Some(Weather::SmokeHaze),
        _ => None,
    }
}

/// Quantize a world position to a 10 m grid cell for stable post
/// keys. Bases don't move, but floating-point roundtrips through
/// objective picks can wobble. Quantizing keeps the same base = same
/// key for the guard-post registry.
pub fn quantize_post_pos(p: [f32; 3]) -> [i32; 3] {
    [
        (p[0] / 10.0).round() as i32,
        0,
        (p[2] / 10.0).round() as i32,
    ]
}
