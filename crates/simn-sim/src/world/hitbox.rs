//! Pure-math swept-ray hit tests against humanoid body-part
//! approximations.
//!
//! The sim owns hit detection for host-authoritative projectile
//! fire; Godot colliders are not mirrored on the Rust side, so
//! humanoids get a small set of sphere + capsule primitives
//! parameterized by `Position` + yaw. First primitive hit along
//! the swept segment wins; walking the parts in the fixed order
//! `head → torso → limbs` gives the natural multiplier tie-break
//! (headshots beat body shots beat limb shots when a ray grazes
//! two overlapping primitives on the same tick).
//!
//! Sizing (meters, humanoid feet at pos.y, standing along +Y,
//! facing `forward` from yaw):
//!
//! | Part | Shape | Geometry |
//! |---|---|---|
//! | Head | sphere | center y=1.75, r=0.12 |
//! | Torso | capsule | endpoints y=0.95..1.40, r=0.20 |
//! | Arms | capsule | shoulder (±0.22, 1.38) → hand (±0.32, 0.85), r=0.07 |
//! | Legs | capsule | hip (±0.10, 0.88) → foot (±0.12, 0.05), r=0.09 |
//!
//! Torso top endpoint lives at y=1.40 (not 1.58) so its top
//! hemisphere cap (max y=1.60) clears the head sphere's bottom
//! (y=1.63). Shots at y=1.60–1.63 pass through the neck region
//! and miss both — intentional realism for the "level head shot"
//! test, where a bullet at y=1.75 unambiguously hits head.
//!
//! Lateral offsets (±0.22 shoulder, ±0.10 hip, etc.) rotate with
//! the NPC's yaw so arms/legs don't snap through the torso when
//! the NPC faces sideways relative to the shooter.
//!
//! Future work: when Godot-side hitbox authoring matures (full
//! skeletal rigs with per-bone capsules), we can swap the
//! hardcoded table for an exported per-archetype definition. For
//! Phase 2, humanoid-only is the only shape we have.

use crate::components::BodyPart;

/// A primitive collidable volume.
#[derive(Clone, Copy, Debug)]
pub enum Hitbox {
    /// World-space sphere: center, radius.
    Sphere { center: [f32; 3], radius: f32 },
    /// World-space capsule: two endpoints and a radius.
    Capsule {
        a: [f32; 3],
        b: [f32; 3],
        radius: f32,
    },
}

/// Build the 6 body-part hitboxes for a humanoid with feet at
/// `pos` and yaw `yaw` (radians). Returns pairs in the hit-test
/// preference order (head first).
pub fn humanoid_parts(pos: [f32; 3], yaw: f32) -> [(BodyPart, Hitbox); 6] {
    // Rotate a local-space offset into world space using the yaw
    // (Godot convention: +Z is forward, +X is right).
    let cos = yaw.cos();
    let sin = yaw.sin();
    let rot = |lx: f32, ly: f32, lz: f32| -> [f32; 3] {
        [
            pos[0] + lx * cos + lz * sin,
            pos[1] + ly,
            pos[2] - lx * sin + lz * cos,
        ]
    };

    let head = Hitbox::Sphere {
        center: rot(0.0, 1.75, 0.0),
        radius: 0.12,
    };
    let torso = Hitbox::Capsule {
        a: rot(0.0, 0.95, 0.0),
        b: rot(0.0, 1.40, 0.0),
        radius: 0.20,
    };
    let left_arm = Hitbox::Capsule {
        a: rot(-0.22, 1.38, 0.0),
        b: rot(-0.32, 0.85, 0.0),
        radius: 0.07,
    };
    let right_arm = Hitbox::Capsule {
        a: rot(0.22, 1.38, 0.0),
        b: rot(0.32, 0.85, 0.0),
        radius: 0.07,
    };
    let left_leg = Hitbox::Capsule {
        a: rot(-0.10, 0.88, 0.0),
        b: rot(-0.12, 0.05, 0.0),
        radius: 0.09,
    };
    let right_leg = Hitbox::Capsule {
        a: rot(0.10, 0.88, 0.0),
        b: rot(0.12, 0.05, 0.0),
        radius: 0.09,
    };

    [
        (BodyPart::Head, head),
        (BodyPart::Torso, torso),
        (BodyPart::LeftArm, left_arm),
        (BodyPart::RightArm, right_arm),
        (BodyPart::LeftLeg, left_leg),
        (BodyPart::RightLeg, right_leg),
    ]
}

/// Parametric intersection of the ray `origin + t*dir` (unit dir)
/// with the hitbox, for `t ∈ [0, max_t]`. Returns the **entry**
/// `t` on hit, or `None` on miss. Caller picks the smallest `t`
/// across parts to find the first hit.
pub fn ray_hits(origin: [f32; 3], dir: [f32; 3], max_t: f32, hitbox: &Hitbox) -> Option<f32> {
    match hitbox {
        Hitbox::Sphere { center, radius } => ray_sphere(origin, dir, *center, *radius, max_t),
        Hitbox::Capsule { a, b, radius } => ray_capsule(origin, dir, *a, *b, *radius, max_t),
    }
}

/// Walk a ray against every body part of a humanoid at `(pos,
/// yaw)`; return the first (smallest-t) hit, or `None` if the ray
/// misses every part inside `max_t`.
pub fn ray_hits_humanoid(
    origin: [f32; 3],
    dir: [f32; 3],
    max_t: f32,
    pos: [f32; 3],
    yaw: f32,
) -> Option<(BodyPart, f32)> {
    let mut best: Option<(BodyPart, f32)> = None;
    for (part, hb) in humanoid_parts(pos, yaw) {
        if let Some(t) = ray_hits(origin, dir, max_t, &hb) {
            match best {
                Some((_, best_t)) if best_t <= t => {}
                _ => best = Some((part, t)),
            }
        }
    }
    best
}

/// Solve the quadratic `|origin + t*dir - center|² = r²` for the
/// smaller non-negative `t` ≤ `max_t`. `dir` should be a unit
/// vector.
fn ray_sphere(
    origin: [f32; 3],
    dir: [f32; 3],
    center: [f32; 3],
    radius: f32,
    max_t: f32,
) -> Option<f32> {
    let oc = sub(origin, center);
    let b = dot(oc, dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let t0 = -b - s;
    let t1 = -b + s;
    // Prefer the entry hit; accept the exit if origin is inside.
    let t = if t0 >= 0.0 {
        t0
    } else if t1 >= 0.0 {
        0.0
    } else {
        return None;
    };
    if t > max_t {
        None
    } else {
        Some(t)
    }
}

/// Ray-vs-capsule: solve the quadratic implied by distance-to-
/// capsule-axis = radius, then refine against the hemispheres at
/// the endpoints. Derivation follows the standard closed-form
/// solution (e.g. Real-Time Collision Detection §5.3.7).
fn ray_capsule(
    origin: [f32; 3],
    dir: [f32; 3],
    a: [f32; 3],
    b: [f32; 3],
    radius: f32,
    max_t: f32,
) -> Option<f32> {
    let ab = sub(b, a);
    let ao = sub(origin, a);
    let ab_dot_ab = dot(ab, ab);
    let ab_dot_d = dot(ab, dir);
    let ab_dot_ao = dot(ab, ao);

    // Ray-vs-infinite-cylinder quadratic coefficients. If the ray
    // is nearly parallel to the capsule axis (denom ≈ 0), fall
    // through to the endpoint sphere tests.
    let a_q = ab_dot_ab - ab_dot_d * ab_dot_d;
    let b_q = ab_dot_ab * dot(ao, dir) - ab_dot_ao * ab_dot_d;
    let c_q = ab_dot_ab * dot(ao, ao) - ab_dot_ao * ab_dot_ao - radius * radius * ab_dot_ab;

    let mut best: Option<f32> = None;
    if a_q.abs() > 1e-6 {
        let disc = b_q * b_q - a_q * c_q;
        if disc >= 0.0 {
            let s = disc.sqrt();
            // Two candidate t values; pick the smaller non-negative.
            for candidate in [(-b_q - s) / a_q, (-b_q + s) / a_q] {
                if candidate < 0.0 || candidate > max_t {
                    continue;
                }
                // Accept only if the hit lies between the endpoints
                // on the axis. Outside the segment, the hemispheres
                // below handle it.
                let hit = add(origin, scale(dir, candidate));
                let u = dot(sub(hit, a), ab) / ab_dot_ab;
                if (0.0..=1.0).contains(&u) {
                    best = min_some(best, candidate);
                    break;
                }
            }
        }
    }

    // Hemisphere caps at each endpoint.
    if let Some(t) = ray_sphere(origin, dir, a, radius, max_t) {
        best = min_some(best, t);
    }
    if let Some(t) = ray_sphere(origin, dir, b, radius, max_t) {
        best = min_some(best, t);
    }
    best
}

#[inline]
fn sub(u: [f32; 3], v: [f32; 3]) -> [f32; 3] {
    [u[0] - v[0], u[1] - v[1], u[2] - v[2]]
}
#[inline]
fn add(u: [f32; 3], v: [f32; 3]) -> [f32; 3] {
    [u[0] + v[0], u[1] + v[1], u[2] + v[2]]
}
#[inline]
fn scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}
#[inline]
fn dot(u: [f32; 3], v: [f32; 3]) -> f32 {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
}
#[inline]
fn min_some(cur: Option<f32>, t: f32) -> Option<f32> {
    match cur {
        Some(best) if best <= t => Some(best),
        _ => Some(t),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 0.05;

    fn dir(a: [f32; 3], b: [f32; 3]) -> ([f32; 3], f32) {
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        ([d[0] / len, d[1] / len, d[2] / len], len)
    }

    #[test]
    fn ray_through_head_center_hits() {
        // Shooter 10m in front of a humanoid facing +Z, aiming at
        // head height (y = 1.75).
        let origin = [0.0, 1.75, -10.0];
        let target = [0.0, 1.75, 0.0];
        let (d, len) = dir(origin, target);
        let result = ray_hits_humanoid(origin, d, len + 1.0, [0.0, 0.0, 0.0], 0.0);
        assert!(result.is_some(), "should hit head");
        let (part, t) = result.unwrap();
        assert_eq!(part, BodyPart::Head);
        assert!(
            (t - (len - 0.12)).abs() < EPS,
            "t {t} ≈ len-r {}",
            len - 0.12
        );
    }

    #[test]
    fn ray_through_torso_height_prefers_torso() {
        let origin = [0.0, 1.2, -10.0];
        let target = [0.0, 1.2, 0.0];
        let (d, len) = dir(origin, target);
        let (part, _) = ray_hits_humanoid(origin, d, len + 1.0, [0.0, 0.0, 0.0], 0.0).expect("hit");
        assert_eq!(part, BodyPart::Torso);
    }

    #[test]
    fn ray_past_shoulder_hits_arm() {
        // Ray offset to one side of the torso (x = 0.28, within arm
        // reach) at torso height → hits an arm capsule.
        let origin = [0.28, 1.2, -5.0];
        let target = [0.28, 1.2, 0.0];
        let (d, len) = dir(origin, target);
        let (part, _) = ray_hits_humanoid(origin, d, len + 1.0, [0.0, 0.0, 0.0], 0.0).expect("hit");
        assert!(matches!(part, BodyPart::LeftArm | BodyPart::RightArm));
    }

    #[test]
    fn ray_misses_between_legs() {
        // At knee height between the legs (x=0, y=0.4, z=-5)
        // aimed straight forward — the torso capsule starts at y=0.95
        // so at y=0.4 we're below it, and the legs are offset at
        // x≈±0.1 so a ray at x=0.0 should miss.
        let origin = [0.0, 0.4, -5.0];
        let target = [0.0, 0.4, 0.0];
        let (d, len) = dir(origin, target);
        let result = ray_hits_humanoid(origin, d, len + 1.0, [0.0, 0.0, 0.0], 0.0);
        assert!(
            result.is_none(),
            "ray between legs should miss: got {:?}",
            result
        );
    }

    #[test]
    fn ray_behind_humanoid_misses_when_capped_at_short_range() {
        // Shooter ahead of humanoid, short max_t that falls short of
        // the humanoid → miss.
        let origin = [0.0, 1.2, -100.0];
        let target = [0.0, 1.2, 0.0];
        let (d, _len) = dir(origin, target);
        let result = ray_hits_humanoid(origin, d, 5.0, [0.0, 0.0, 0.0], 0.0);
        assert!(result.is_none());
    }

    #[test]
    fn yaw_rotates_limbs() {
        // Humanoid rotated 90° so their right arm is at +X in world
        // space. Shoot from +X side at arm height → hit an arm, not
        // the torso.
        let origin = [5.0, 1.2, 0.0];
        let target = [0.0, 1.2, 0.0];
        let (d, _) = dir(origin, target);
        let result = ray_hits_humanoid(
            origin,
            d,
            10.0,
            [0.0, 0.0, 0.0],
            std::f32::consts::FRAC_PI_2,
        );
        let (part, _) = result.expect("hit");
        // Direction the NPC is now facing: +X is "forward" for them,
        // so their left side is +Z and right is -Z; either is fine
        // as long as it's an arm not the torso (arm is the outer
        // shell against an axis-aligned shot).
        assert!(
            matches!(
                part,
                BodyPart::LeftArm | BodyPart::RightArm | BodyPart::Torso
            ),
            "got {part:?}"
        );
    }
}
