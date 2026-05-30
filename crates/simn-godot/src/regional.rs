//! [`RegionalBackdrop`] — distant-terrain backdrop mesh that
//! envelopes the playable map. Loads the shared `_regional`
//! heightmap (~80 km × 110 km of the Columbia Gorge at 100 m sample
//! spacing) and positions it geographically correct relative to
//! whatever playable map is currently in the scene, so the level
//! edge reads as continuous landscape instead of a hard cliff.
//!
//! On `ready`:
//!
//! 1. Loads `res://assets/terrain/_regional/` (heightmap +
//!    features).
//! 2. Walks siblings to find the playable `TerrainNode`; reads its
//!    `map_id` export.
//! 3. Loads the playable map's `terrain.toml` for its UTM origin.
//! 4. Builds an `ArrayMesh` from the regional heightmap, **skipping
//!    any quad inside the playable map's bounds** (hole-punch — no
//!    z-fighting with the foreground terrain).
//! 5. Positions itself in world space such that the regional UTM
//!    origin maps to the playable scene's coordinate frame.
//! 6. Wires the regional shader's `features_texture`, `diffuse_array`,
//!    and `terrain_extent_m` uniforms.
//!
//! Non-collidable. Renders with `terrain_regional.gdshader` —
//! ~5-8 texture samples per fragment vs ~96 for the playable
//! shader. Frustum-culled by Godot, fog-faded by the shared
//! WorldEnvironment.

use godot::classes::image::Format as ImageFormat;
use godot::classes::mesh::{ArrayType, PrimitiveType};
use godot::classes::{
    ArrayMesh, IMeshInstance3D, Image, ImageTexture, MeshInstance3D, Node, ProjectSettings,
    ShaderMaterial,
};
use godot::prelude::*;
use simn_terrain::{Heightmap, TerrainMetadata};
use std::path::PathBuf;

const REGIONAL_MATERIAL_PATH: &str = "res://resources/materials/terrain_regional.tres";

#[derive(GodotClass)]
#[class(tool, base=MeshInstance3D)]
pub struct RegionalBackdrop {
    /// Logical map id of the regional asset under
    /// `res://assets/terrain/`. Default `_regional`.
    #[export]
    regional_map_id: GString,

    /// Pre-populated to `terrain_regional.tres`. Inspector edits
    /// flow through to the running mesh — same pattern as
    /// `TerrainNode.terrain_material`.
    #[export]
    regional_material: Option<Gd<ShaderMaterial>>,

    base: Base<MeshInstance3D>,
}

#[godot_api]
impl IMeshInstance3D for RegionalBackdrop {
    fn init(base: Base<MeshInstance3D>) -> Self {
        Self {
            regional_map_id: GString::from("_regional"),
            regional_material: try_load::<ShaderMaterial>(REGIONAL_MATERIAL_PATH).ok(),
            base,
        }
    }

    fn ready(&mut self) {
        if let Err(e) = self.try_build() {
            godot_warn!("RegionalBackdrop: {e}");
        }
    }
}

impl RegionalBackdrop {
    fn try_build(&mut self) -> Result<(), String> {
        // 1. Load regional heightmap.
        let regional_id = self.regional_map_id.to_string();
        let regional_dir =
            globalize_terrain_path(&regional_id).ok_or("regional asset path not found")?;
        let regional_hm = Heightmap::load(&regional_dir)
            .map_err(|e| format!("loading regional heightmap: {e}"))?;
        let regional_meta = regional_hm.metadata().clone();

        // 2. Find the playable TerrainNode sibling, read its map_id
        //    via property reflection (not a typed cast — keeps
        //    regional decoupled from terrain.rs internals).
        let playable_id = playable_map_id_from_siblings(&self.base().clone().upcast::<Node>())
            .ok_or("no sibling TerrainNode with `map_id` export found")?;

        // 3. Load the playable map's metadata. We don't load the
        //    full heightmap — we only need the UTM origin + extent.
        let playable_dir =
            globalize_terrain_path(&playable_id).ok_or("playable map asset path not found")?;
        let playable_meta = TerrainMetadata::load(&playable_dir)
            .map_err(|e| format!("loading playable metadata: {e}"))?;

        // 4. Compute the playable's bounding rect in **regional-
        //    local meters** (offset from the regional NW corner).
        //    Used for hole-punching the mesh.
        let playable_extent = playable_meta.extent_m();
        let utm_offset_east = playable_meta.origin_utm_easting - regional_meta.origin_utm_easting;
        let utm_offset_north =
            regional_meta.origin_utm_northing - playable_meta.origin_utm_northing;
        let skip_x_min = utm_offset_east as f32;
        let skip_z_min = utm_offset_north as f32;
        let skip_x_max = skip_x_min + playable_extent[0];
        let skip_z_max = skip_z_min + playable_extent[1];

        // 5. Build the ArrayMesh.
        let mesh = build_regional_mesh(
            &regional_hm,
            (skip_x_min, skip_z_min, skip_x_max, skip_z_max),
        );

        // 6. Position the mesh so the regional UTM origin lines up
        //    with the playable scene's world origin.
        //
        //    The playable terrain is centered on its scene root
        //    (TerrainNode at (0,0,0), mesh runs from -extent/2 to
        //    +extent/2). The regional mesh is *not* centered — its
        //    NW corner is at (0, 0) in the mesh's local frame.
        //
        //    We want the playable's world position (0,0,0) to
        //    correspond to (UTM east_p, north_p) in geo space.
        //    The regional NW (0,0) in mesh frame corresponds to
        //    UTM (east_r, north_r). Offset between them:
        //
        //        Δ_east  = east_p - east_r  (m east)
        //        Δ_north = north_p - north_r (m north, positive = N)
        //
        //    In Godot world coords (X = east, Z = south):
        //        backdrop position.x = -Δ_east - extent_p_x / 2
        //        backdrop position.z = +Δ_north - extent_p_z / 2
        //
        //    (The -extent_p/2 shifts because the playable mesh is
        //    centered, not anchored at NW.)
        let playable_extent = playable_meta.extent_m();
        let pos_x = -(utm_offset_east as f32) - playable_extent[0] * 0.5;
        let pos_z = -(utm_offset_north as f32) - playable_extent[1] * 0.5;
        self.base_mut()
            .set_position(Vector3::new(pos_x, 0.0, pos_z));

        // 7. Wire the shader uniforms (features texture + diffuse
        //    array + extent).
        let Some(mut mat) = self.regional_material.clone() else {
            self.base_mut()
                .set_mesh(&mesh.upcast::<godot::classes::Mesh>());
            godot_warn!("RegionalBackdrop: no material set; mesh will render with default");
            return Ok(());
        };

        if let Some(features) = regional_hm.features_bytes() {
            let w = regional_meta.width as i32;
            let h = regional_meta.height as i32;
            if let Some(tex) = build_r8_texture(features, w, h) {
                mat.set_shader_parameter("features_texture", &tex.to_variant());
            }
        }
        if let Some(arr) =
            crate::terrain::build_diffuse_array(&crate::terrain::TERRAIN_CLASS_DIFFUSE, "regional")
        {
            mat.set_shader_parameter("diffuse_array", &arr.to_variant());
        }
        let [ex, ez] = regional_hm.extent_m();
        mat.set_shader_parameter("terrain_extent_m", &Vector2::new(ex, ez).to_variant());

        self.base_mut()
            .set_material_override(&mat.upcast::<godot::classes::Material>());
        self.base_mut()
            .set_mesh(&mesh.upcast::<godot::classes::Mesh>());

        // Backdrop never casts shadows on the playable terrain
        // (low-res mesh would create visible shadow seams) and
        // shouldn't receive shadows from far-away sources.
        self.base_mut().set_cast_shadows_setting(
            godot::classes::geometry_instance_3d::ShadowCastingSetting::OFF,
        );
        Ok(())
    }
}

/// Resolve `res://assets/terrain/<map_id>` to an OS path via
/// `ProjectSettings::globalize_path`.
fn globalize_terrain_path(map_id: &str) -> Option<PathBuf> {
    let res = format!("res://assets/terrain/{map_id}");
    let globalized = ProjectSettings::singleton().globalize_path(&GString::from(&res));
    let s = globalized.to_string();
    if s.is_empty() {
        None
    } else {
        Some(PathBuf::from(s))
    }
}

/// Walk the parent's children looking for a `TerrainNode` with a
/// `map_id` export. Returns the resolved string id. Skips the
/// `RegionalBackdrop` itself.
fn playable_map_id_from_siblings(self_node: &Gd<Node>) -> Option<String> {
    let parent = self_node.get_parent()?;
    let n = parent.get_child_count();
    for i in 0..n {
        let Some(child) = parent.get_child(i) else {
            continue;
        };
        if child == *self_node {
            continue;
        }
        // We don't import TerrainNode here to keep regional
        // decoupled — read by class-name + property name.
        let class_name: StringName = (&child.get_class()).into();
        #[allow(clippy::cmp_owned)]
        let is_terrain = class_name == StringName::from("TerrainNode");
        if is_terrain {
            let v = child.get("map_id");
            let s: GString = v.try_to().ok()?;
            let s = s.to_string();
            if !s.is_empty() {
                return Some(s);
            }
        }
    }
    None
}

/// Build a single-channel R8 ImageTexture from raw class bytes.
fn build_r8_texture(bytes: &[u8], w: i32, h: i32) -> Option<Gd<godot::classes::Texture2D>> {
    let mut pba = PackedByteArray::new();
    pba.resize(bytes.len());
    for (i, &b) in bytes.iter().enumerate() {
        pba[i] = b;
    }
    let img = Image::create_from_data(w, h, false, ImageFormat::R8, &pba)?;
    let tex = ImageTexture::create_from_image(&img)?;
    Some(tex.upcast::<godot::classes::Texture2D>())
}

/// Build a triangle mesh from the regional heightmap. Optionally
/// skip any quad whose all-4 corners fall inside the playable
/// `(x_min, z_min, x_max, z_max)` rectangle in regional-local
/// meters (NW corner = (0, 0)).
fn build_regional_mesh(hm: &Heightmap, skip: (f32, f32, f32, f32)) -> Gd<ArrayMesh> {
    let w = hm.width() as usize;
    let h = hm.height() as usize;
    let spacing = hm.metadata().spacing_m;
    let (skip_xmin, skip_zmin, skip_xmax, skip_zmax) = skip;

    // Vertex/normal/uv for every grid sample.
    let mut verts = PackedVector3Array::new();
    let mut normals = PackedVector3Array::new();
    let mut uvs = PackedVector2Array::new();
    verts.resize(w * h);
    normals.resize(w * h);
    uvs.resize(w * h);

    // Vertices inside the playable's hole rect get dropped 200 m so
    // the playable mesh's edge always hangs over the regional —
    // closes the visible "peek-under" gap that comes from heightmap
    // resolution mismatch (regional 100 m vs playable 2 m, Y at the
    // shared boundary disagrees by 5-50 m typically). The drop only
    // affects the boundary-adjacent quads (since interior-of-hole
    // quads are skipped entirely below); they form a "moat" lip
    // pulling the regional well below the playable surface.
    const HOLE_MOAT_DROP_M: f32 = 200.0;

    for row in 0..h {
        for col in 0..w {
            let x = col as f32 * spacing;
            let z = row as f32 * spacing;
            let mut y = hm.sample(x, z);
            let inside_hole = x >= skip_xmin && x <= skip_xmax && z >= skip_zmin && z <= skip_zmax;
            if inside_hole {
                y -= HOLE_MOAT_DROP_M;
            }
            let n = hm.sample_normal(x, z);
            let i = row * w + col;
            verts[i] = Vector3::new(x, y, z);
            normals[i] = Vector3::new(n[0], n[1], n[2]);
            uvs[i] = Vector2::new(col as f32 / (w - 1) as f32, row as f32 / (h - 1) as f32);
        }
    }

    // Triangle indices, skipping quads inside the playable rect.
    let mut indices = PackedInt32Array::new();
    for row in 0..(h - 1) {
        for col in 0..(w - 1) {
            // Quad corners in regional-local meters.
            let qx_min = col as f32 * spacing;
            let qz_min = row as f32 * spacing;
            let qx_max = qx_min + spacing;
            let qz_max = qz_min + spacing;
            // Skip if entirely inside the playable rect (any of the
            // 4 corners outside → keep the quad, so there's a
            // 1-cell skirt around the hole).
            if qx_min >= skip_xmin
                && qx_max <= skip_xmax
                && qz_min >= skip_zmin
                && qz_max <= skip_zmax
            {
                continue;
            }
            let i00 = (row * w + col) as i32;
            let i10 = (row * w + col + 1) as i32;
            let i01 = ((row + 1) * w + col) as i32;
            let i11 = ((row + 1) * w + col + 1) as i32;
            // Two triangles per quad. Winding is CCW when viewed
            // from above (camera looking down) so Godot's cull_back
            // keeps front-faces visible from above. Got this wrong
            // initially — first revision was CW-from-above which
            // backface-culled the entire mesh except for steep
            // slopes that happened to face the camera, leaving
            // ribbon-shaped artifacts along ridges.
            //
            // Walking NW → NE → SW in world XZ (X=east, Z=south)
            // traces top-left → top-right → bottom-left, which is
            // CCW projected onto the XZ plane viewed from +Y.
            indices.push(i00);
            indices.push(i10);
            indices.push(i01);
            indices.push(i10);
            indices.push(i11);
            indices.push(i01);
        }
    }

    let mut arrays = VarArray::new();
    arrays.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
    arrays.set(ArrayType::VERTEX.ord() as usize, &verts.to_variant());
    arrays.set(ArrayType::NORMAL.ord() as usize, &normals.to_variant());
    arrays.set(ArrayType::TEX_UV.ord() as usize, &uvs.to_variant());
    arrays.set(ArrayType::INDEX.ord() as usize, &indices.to_variant());

    let mut mesh = ArrayMesh::new_gd();
    mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &arrays);
    mesh
}
