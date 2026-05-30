//! Runtime physics-shape population for static prop scenes.
//!
//! Static prop scenes are expected to be shaped as:
//! ```text
//! StaticBody3D (Model)
//!   ├─ MeshInstance3D
//!   └─ CollisionShape3D  (shape = null)
//! ```
//!
//! The `CollisionShape3D` is intentionally empty because baking a
//! `ConcavePolygonShape3D` sub-resource into every prop `.tscn` would
//! double on-disk size and slow scene loads. Instead, game code calls
//! `PhysicsSetup.attach_static_collision(model)` on spawned props, which
//! reads the sibling `MeshInstance3D`'s mesh and calls
//! `ArrayMesh::create_trimesh_shape()` to build the collision shape
//! lazily at runtime.
//!
//! For hot code paths you can cache the resulting shapes per-mesh so that
//! spawning 200 instances of the same prop builds the shape once.

use godot::classes::{CollisionShape3D, ConcavePolygonShape3D, MeshInstance3D};
use godot::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static SHAPE_CACHE: RefCell<HashMap<String, Gd<ConcavePolygonShape3D>>> =
        RefCell::new(HashMap::new());
}

fn with_cache<F, R>(f: F) -> R
where
    F: FnOnce(&mut HashMap<String, Gd<ConcavePolygonShape3D>>) -> R,
{
    SHAPE_CACHE.with(|c| f(&mut c.borrow_mut()))
}

#[derive(GodotClass)]
#[class(init, base=RefCounted)]
pub struct PhysicsSetup {
    base: Base<RefCounted>,
}

#[godot_api]
impl PhysicsSetup {
    /// Populate empty `CollisionShape3D` children of a static prop with
    /// trimesh shapes built from their sibling `MeshInstance3D` meshes.
    ///
    /// Call this on any spawned instance of a static prop scene before the
    /// physics server needs collision. Returns the number of shapes that
    /// were populated (including cached hits).
    #[func]
    pub fn attach_static_collision(&mut self, model: Gd<Node>) -> i32 {
        let mut count = 0i32;

        // Find the mesh: first MeshInstance3D child of `model`.
        let mesh_instance = find_child_of_type::<MeshInstance3D>(&model);
        let Some(mi) = mesh_instance else {
            return 0;
        };
        let Some(mesh) = mi.get_mesh() else {
            return 0;
        };

        // Cache key: mesh resource path. If the mesh was loaded from
        // res://... the path is stable; runtime-built meshes have an
        // empty path and fall through to build-each-time.
        let cache_key = mesh.get_path().to_string();

        let shape: Gd<ConcavePolygonShape3D> = if cache_key.is_empty() {
            // No stable key, build without caching.
            match mesh.create_trimesh_shape() {
                Some(s) => s,
                None => return 0,
            }
        } else {
            match with_cache(|cache| cache.get(&cache_key).cloned()) {
                Some(cached) => cached,
                None => {
                    let built = match mesh.create_trimesh_shape() {
                        Some(s) => s,
                        None => return 0,
                    };
                    with_cache(|cache| cache.insert(cache_key, built.clone()));
                    built
                }
            }
        };

        // Assign to every CollisionShape3D child whose shape is null.
        for i in 0..model.get_child_count() {
            if let Some(child) = model.get_child(i) {
                if let Ok(mut cs) = child.try_cast::<CollisionShape3D>() {
                    if cs.get_shape().is_none() {
                        cs.set_shape(&shape);
                        count += 1;
                    }
                }
            }
        }

        count
    }

    /// Clear the process-wide shape cache. Useful when hot-reloading assets.
    #[func]
    pub fn clear_cache(&mut self) {
        with_cache(|cache| cache.clear());
    }
}

fn find_child_of_type<T: Inherits<Node>>(parent: &Gd<Node>) -> Option<Gd<T>> {
    for i in 0..parent.get_child_count() {
        if let Some(child) = parent.get_child(i) {
            if let Ok(typed) = child.try_cast::<T>() {
                return Some(typed);
            }
        }
    }
    None
}
