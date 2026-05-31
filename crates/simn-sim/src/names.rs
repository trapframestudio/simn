//! Procedural name registry. Multicultural name pools loaded at
//! startup from per-bucket text files (one name per line). Every
//! NPC, regardless of faction, draws from the same global
//! distribution — factions in the setting are demographically mixed,
//! and the rolled bucket implies the NPC's ethnic background, which
//! later drives character-mesh selection.
//!
//! # Sources + expansion
//!
//! Initial pools are hand-curated samples (~50–60 first + ~50–60
//! last per bucket) drawn from widely-known public names. To expand,
//! drop additional names into
//! `crates/simn-sim/content/names/{first,last}/{bucket}.txt`, one per
//! line — they're loaded via `include_str!` at compile time. The
//! `smashew/NameDatabases` repo on GitHub (MIT) is a good upstream
//! source for thousands more per bucket; this module's loader is
//! format-compatible (one-name-per-line) so vendoring is mechanical.
//!
//! # Coverage
//!
//! 8 buckets at ship: American (English / mixed-European), Latin
//! American (Spanish), Slavic (Russian + Eastern European), East
//! Asian (Chinese / pinyin), South Asian (Indian), West African,
//! Western European (Germanic / Romance / Nordic), Middle Eastern
//! (Arabic). Add more buckets via the `NationalityBucket` enum +
//! matching files; the loader walks `ALL`.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

/// Cultural / ethnic-origin bucket for a procedurally-rolled NPC
/// name. Drives both name selection and (downstream) character
/// mesh-set selection. Uniformly weighted by default; per-faction
/// overrides can come later if any faction needs a distinct
/// demographic mix.
#[repr(u8)]
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum NationalityBucket {
    American,
    LatinAmerican,
    Slavic,
    EastAsian,
    SouthAsian,
    WestAfrican,
    WesternEuropean,
    MiddleEastern,
}

impl NationalityBucket {
    /// All buckets in canonical iteration order. Used by tests and
    /// the registry loader.
    pub const ALL: [NationalityBucket; 8] = [
        NationalityBucket::American,
        NationalityBucket::LatinAmerican,
        NationalityBucket::Slavic,
        NationalityBucket::EastAsian,
        NationalityBucket::SouthAsian,
        NationalityBucket::WestAfrican,
        NationalityBucket::WesternEuropean,
        NationalityBucket::MiddleEastern,
    ];

    /// Snake-case identifier matching the data file basename.
    pub fn name(self) -> &'static str {
        match self {
            Self::American => "american",
            Self::LatinAmerican => "latin_american",
            Self::Slavic => "slavic",
            Self::EastAsian => "east_asian",
            Self::SouthAsian => "south_asian",
            Self::WestAfrican => "west_african",
            Self::WesternEuropean => "western_european",
            Self::MiddleEastern => "middle_eastern",
        }
    }

    /// Inverse of [`Self::name`]: parse a snake-case bucket tag.
    /// Returns `None` for unknown tags so faction TOMLs can carry
    /// a typo without panicking the loader (the bucket is just
    /// dropped from that faction's weight list).
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "american" => Some(Self::American),
            "latin_american" => Some(Self::LatinAmerican),
            "slavic" => Some(Self::Slavic),
            "east_asian" => Some(Self::EastAsian),
            "south_asian" => Some(Self::SouthAsian),
            "west_african" => Some(Self::WestAfrican),
            "western_european" => Some(Self::WesternEuropean),
            "middle_eastern" => Some(Self::MiddleEastern),
            _ => None,
        }
    }

    /// Default global weight when picking a bucket for a new NPC.
    /// All buckets are weighted equally for now; tune later if the
    /// world's demographic mix shifts. Per-faction overrides are an
    /// open extension point (add a `nationality_weights` field to
    /// `factions.toml` when needed).
    pub fn default_weight(self) -> u32 {
        1
    }
}

/// One bucket's first + last name lists. First names are split
/// by gender (male / female) via the `M `/`F ` line prefix in
/// the per-bucket text files; untagged lines fall into the male
/// pool by default (matches the male-skewed combatant population
/// the setting assumes). Last names aren't gendered.
#[derive(Clone)]
struct BucketPool {
    bucket: NationalityBucket,
    first_male: Vec<String>,
    first_female: Vec<String>,
    last: Vec<String>,
}

/// Default male/female split for new NPC names. The setting
/// assumes a male-skewed combatant population — nomads,
/// soldiers, faction recruits — so the default name draw is
/// heavily weighted male. Factions can override per their own
/// `male_name_weight` knob (e.g. medical / civilian factions
/// with mixed gender). Range `[0.0, 1.0]`; 0 = all female,
/// 1 = all male.
pub const DEFAULT_MALE_NAME_WEIGHT: f32 = 0.92;

/// Resource holding all name pools. Loaded once at sim startup;
/// read-only thereafter.
#[derive(Resource, Clone)]
pub struct NameRegistry {
    pools: Vec<BucketPool>,
}

impl NameRegistry {
    /// Build the canonical registry from compiled-in data files. The
    /// per-bucket text files live under
    /// `crates/simn-sim/content/names/{first,last}/{bucket}.txt`. Each
    /// is one name per line; blank lines and `#` comments are
    /// skipped. Cached process-wide via `OnceLock`; see
    /// [`crate::items::ItemRegistry::load`] for the rationale.
    pub fn load() -> Self {
        static CACHE: std::sync::OnceLock<NameRegistry> = std::sync::OnceLock::new();
        CACHE
            .get_or_init(|| Self::parse_from_data(&crate::ContentSource::Embedded))
            .clone()
    }

    /// Load from an explicit content source; see [`crate::items::ItemRegistry::load_from`].
    pub fn load_from(src: &crate::ContentSource) -> Self {
        match src {
            crate::ContentSource::Embedded => Self::load(),
            other => Self::parse_from_data(other),
        }
    }

    fn parse_from_data(src: &crate::ContentSource) -> Self {
        let read = |path: String| {
            src.read_str(&path)
                .unwrap_or_else(|e| panic!("names content load failed: {e}"))
        };
        let pools: Vec<BucketPool> = NationalityBucket::ALL
            .iter()
            .map(|&b| {
                let (first_male, first_female) = parse_first_names(&read(first_path(b)));
                BucketPool {
                    bucket: b,
                    first_male,
                    first_female,
                    last: parse_names(&read(last_path(b))),
                }
            })
            .collect();
        Self { pools }
    }

    /// Pick a bucket weighted by `default_weight`, then pick a first
    /// and last name from that bucket via `rng`. Returns
    /// `(bucket, "First Last")`. Deterministic from `rng`. Uses
    /// [`DEFAULT_MALE_NAME_WEIGHT`] for the gender split.
    pub fn roll<R: rand::Rng>(&self, rng: &mut R) -> (NationalityBucket, String) {
        let weights: Vec<(NationalityBucket, u32)> = self
            .pools
            .iter()
            .map(|p| (p.bucket, p.bucket.default_weight()))
            .collect();
        self.roll_with_weights(rng, &weights, DEFAULT_MALE_NAME_WEIGHT)
    }

    /// Pick a bucket via a per-call weighted distribution, then a
    /// first + last name from that bucket. Empty `weights` (or all
    /// zeros) falls back to a uniform draw across all buckets so
    /// a missing / mistyped faction TOML never panics.
    ///
    /// `male_weight` is the probability `[0.0, 1.0]` of drawing
    /// from the male first-name pool; `1.0 - male_weight` falls
    /// to the female pool. The setting assumes a male-skewed
    /// combatant population so the default is
    /// [`DEFAULT_MALE_NAME_WEIGHT`]; factions can override per
    /// their `male_name_weight` in `factions.toml`.
    pub fn roll_with_weights<R: rand::Rng>(
        &self,
        rng: &mut R,
        weights: &[(NationalityBucket, u32)],
        male_weight: f32,
    ) -> (NationalityBucket, String) {
        // Sum non-zero weighted entries that we recognize.
        let total: u32 = weights.iter().map(|(_, w)| *w).sum();
        let bucket = if total == 0 {
            // Fallback: uniform across ALL buckets.
            let n = NationalityBucket::ALL.len();
            NationalityBucket::ALL[rng.gen_range(0..n)]
        } else {
            let mut roll = rng.gen_range(0..total);
            let mut chosen = weights[0].0;
            for (b, w) in weights {
                if *w == 0 {
                    continue;
                }
                if roll < *w {
                    chosen = *b;
                    break;
                }
                roll -= *w;
            }
            chosen
        };
        let pool = self
            .pools
            .iter()
            .find(|p| p.bucket == bucket)
            .expect("ALL buckets present in registry");
        // Gender pick: draw from male pool with prob `male_weight`,
        // female pool otherwise. Either pool empty falls through to
        // the other (so a bucket with no female names tagged still
        // works — common for newly-added buckets pre-tagging).
        let male = rng.gen_range(0.0..1.0_f32) < male_weight.clamp(0.0, 1.0);
        let first_pool = if male {
            if pool.first_male.is_empty() {
                &pool.first_female
            } else {
                &pool.first_male
            }
        } else if pool.first_female.is_empty() {
            &pool.first_male
        } else {
            &pool.first_female
        };
        let first = first_pool
            .get(rng.gen_range(0..first_pool.len().max(1)))
            .cloned()
            .unwrap_or_else(|| "Anon".to_string());
        let last = pool
            .last
            .get(rng.gen_range(0..pool.last.len().max(1)))
            .cloned()
            .unwrap_or_else(|| "Doe".to_string());
        (bucket, format!("{first} {last}"))
    }

    /// Roll using a faction's `nationality_weights` map (string-keyed
    /// per the TOML schema). Unrecognized tags are dropped silently;
    /// an empty map (the common case) falls back to a uniform draw.
    /// Uses [`DEFAULT_MALE_NAME_WEIGHT`] for the gender split when
    /// the faction doesn't carry an override; callers with access
    /// to the `FactionDef` should prefer
    /// [`Self::roll_for_faction_gendered`].
    pub fn roll_for_faction<R: rand::Rng>(
        &self,
        rng: &mut R,
        faction_weights: &std::collections::HashMap<String, u32>,
    ) -> (NationalityBucket, String) {
        self.roll_for_faction_gendered(rng, faction_weights, DEFAULT_MALE_NAME_WEIGHT)
    }

    /// Same as [`Self::roll_for_faction`] but accepts the
    /// per-faction `male_name_weight` override. Default in
    /// `factions.toml` is omitted ⇒ caller passes
    /// [`DEFAULT_MALE_NAME_WEIGHT`].
    pub fn roll_for_faction_gendered<R: rand::Rng>(
        &self,
        rng: &mut R,
        faction_weights: &std::collections::HashMap<String, u32>,
        male_weight: f32,
    ) -> (NationalityBucket, String) {
        let weights: Vec<(NationalityBucket, u32)> = faction_weights
            .iter()
            .filter_map(|(k, w)| NationalityBucket::from_name(k).map(|b| (b, *w)))
            .collect();
        let weights = if weights.is_empty() {
            self.pools
                .iter()
                .map(|p| (p.bucket, p.bucket.default_weight()))
                .collect::<Vec<_>>()
        } else {
            weights
        };
        self.roll_with_weights(rng, &weights, male_weight)
    }

    /// Accessors for tests / debug. Returns the per-bucket first
    /// or last list. `first_names` returns the union of male +
    /// female pools (matches the pre-gendered API for existing
    /// callers); use `first_names_male` / `first_names_female`
    /// for the split.
    pub fn first_names(&self, bucket: NationalityBucket) -> Vec<&str> {
        let Some(pool) = self.pools.iter().find(|p| p.bucket == bucket) else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(pool.first_male.len() + pool.first_female.len());
        out.extend(pool.first_male.iter().map(String::as_str));
        out.extend(pool.first_female.iter().map(String::as_str));
        out
    }

    pub fn first_names_male(&self, bucket: NationalityBucket) -> &[String] {
        self.pools
            .iter()
            .find(|p| p.bucket == bucket)
            .map(|p| p.first_male.as_slice())
            .unwrap_or(&[])
    }

    pub fn first_names_female(&self, bucket: NationalityBucket) -> &[String] {
        self.pools
            .iter()
            .find(|p| p.bucket == bucket)
            .map(|p| p.first_female.as_slice())
            .unwrap_or(&[])
    }

    pub fn last_names(&self, bucket: NationalityBucket) -> &[String] {
        self.pools
            .iter()
            .find(|p| p.bucket == bucket)
            .map(|p| p.last.as_slice())
            .unwrap_or(&[])
    }
}

fn parse_names(raw: &str) -> Vec<String> {
    raw.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// Parse a first-name file with optional `M `/`F ` gender
/// prefixes. Untagged lines fall into the male pool (matches
/// the setting's male-skewed combatant population — a name
/// added without a prefix is more likely male than not, so
/// this is the safer default).
///
/// Returns owned `(male, female)` name vectors. Both may be empty
/// if the file is blank, in which case the registry's roll path
/// falls back to "Anon" via `unwrap_or` on `gen_range(0..0)`.
fn parse_first_names(raw: &str) -> (Vec<String>, Vec<String>) {
    let mut male: Vec<String> = Vec::new();
    let mut female: Vec<String> = Vec::new();
    for raw_line in raw.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Strip a single-letter gender prefix `M ` / `F ` (case-
        // insensitive); untagged lines fall into the male pool.
        let (gender, name) = match line.as_bytes() {
            [b'M' | b'm', b' ', ..] => ('m', line[2..].trim()),
            [b'F' | b'f', b' ', ..] => ('f', line[2..].trim()),
            _ => ('m', line),
        };
        match gender {
            'm' => male.push(name.to_string()),
            'f' => female.push(name.to_string()),
            _ => unreachable!(),
        }
    }
    (male, female)
}

/// Stable file stem for a bucket, used to build content logical
/// paths (`names/first/<stem>.txt`, `names/last/<stem>.txt`).
fn bucket_file(b: NationalityBucket) -> &'static str {
    match b {
        NationalityBucket::American => "american",
        NationalityBucket::LatinAmerican => "latin_american",
        NationalityBucket::Slavic => "slavic",
        NationalityBucket::EastAsian => "east_asian",
        NationalityBucket::SouthAsian => "south_asian",
        NationalityBucket::WestAfrican => "west_african",
        NationalityBucket::WesternEuropean => "western_european",
        NationalityBucket::MiddleEastern => "middle_eastern",
    }
}

fn first_path(b: NationalityBucket) -> String {
    format!("names/first/{}.txt", bucket_file(b))
}

fn last_path(b: NationalityBucket) -> String {
    format!("names/last/{}.txt", bucket_file(b))
}
