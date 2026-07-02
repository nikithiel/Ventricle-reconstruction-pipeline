bl_info = {
    "name" : "Geometrical heart reconstrucion", 
    "author" : "Daniel Verhuelsdonk",
    "version" : (2, 1),
    "blender" : (3, 1, 0),
    "location" : "Operator Search",
    "description": "Panel and operators to geometrically reconstruct the upper heart shape",
    "warning" : "",
    "wiki_url" : "",
    "category" : "Add Mesh",
}
# Imports
import bpy
import numba as nb
import numpy as np
import bmesh
import math
import mathutils 
import open3d as o3d
import os
import re

from pycpd import RigidRegistration
from mathutils.bvhtree import BVHTree
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import multiprocessing as mp

from pathlib import Path


from stl_plot import check_stl_and_connectivity, derive_ed_es_from_volume_curve, parseRunTimeVariables_unique
from calculate_valve_diameters import main_calc_diameter
scene = bpy.types.Scene

dev_env_tools = True

# Generally used functions.
def cons_print(data):
    """Print to console for button presses. Used for error messages, information outputs and warnings"""
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'CONSOLE':
                override = {'window': window, 'screen': screen, 'area': area}
                bpy.ops.console.scrollback_append(override, text=str(data), type="OUTPUT")   

def copy_object(input_name, output_name):
    """Copy object with given name"""
    src_obj = bpy.data.objects[input_name]
    new_obj = src_obj.copy()
    new_obj.data = src_obj.data.copy()
    new_obj.data.name = output_name 
    new_obj.animation_data_clear()
    new_obj.name = output_name
    bpy.context.collection.objects.link(new_obj)
    return new_obj

def deselect_object_vertices(obj):
    """Go into edit mode and deselect all vertices of an object"""
    # Transfer data into edit mode.
    me = obj.data
    bpy.ops.object.mode_set(mode='EDIT') 
    bm = bmesh.from_edit_mesh(me)
    # Deselect all vertices.
    for v in bm.verts: v.select = False
    # Return to object mode and update the mesh to the obeject.
    bm.select_flush_mode()   
    me.update()
    bpy.ops.object.mode_set(mode='OBJECT') 

def join_objects(obj, joined_obj):
    """Join two objects without changing the selection"""
    # Set correct selections before joining geometries.
    bpy.ops.object.mode_set(mode='OBJECT') 
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    joined_obj.select_set(True)
    bpy.ops.object.mode_set(mode='OBJECT') 
    bpy.ops.object.join() # Join objects.

def get_selected_edges(obj):
    """Return currenctly selected edges and corresponding vertices of active object"""
    mesh = bmesh.from_edit_mesh(obj.data)
    active_edges_verts, active_edges_indices = ([] for i in range(2))
    for e in mesh.edges:
        if e.select:
            active_edges_verts.append((e.verts[0].index, e.verts[1].index))
            active_edges_indices.append(e.index)
    return active_edges_indices, active_edges_verts

def list_diff(list1, list2):
    """Difference between two lists"""
    li_dif = [i for i in list1 + list2 if i not in list1 or i not in list2]
    return li_dif

def transfer_data_to_mesh(obj):
    """Transfer input object data to mesh and return bmesh object"""
    bm = bmesh.new()       
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    return bm

def get_value(self):
    return "//"

def set_value(self,value):
    self["ventricle_import_dir"] = value
# UI-interface functions.
class MESH_OT_get_node(bpy.types.Operator):
    """Get node position coordinates and save coordinates in UI"""
    bl_idname = 'heart.get_point'
    bl_label = 'Get node position coordinates and save coordinates in UI.' 
    point_mode: bpy.props.StringProperty(name = "point_mode", description="Which point to select", default = "Top") # Which point is selected with this function(Top/basal, bottom/apical, septum)
    def execute(self, context):
        obj = context.object
        if obj.mode != 'EDIT': # Exception when in object mode.
            cons_print('This process only works in edit mode.')
            return{'CANCELLED'} 
        # Get vector-coordinates.
        bm = bmesh.from_edit_mesh(obj.data)
        counter = 0
        for v in bm.verts:
            if v.select:
                counter += 1 # Increment to get the number of selected nodes
                vertice_coords = obj.matrix_world @ v.co # Transform to global coordinate system
        # Check for the correct number of nodes selected.
        if counter !=1:
            cons_print('Incorrect amount of nodes. Only one node may be selected.')
            return{'CANCELLED'}  
        # Using temporary veriables prevents the change of the global variable, when to many nodes are selected.
        if self.point_mode == "Top":  
            context.scene.pos_top = vertice_coords
        elif self.point_mode == "Bot": context.scene.pos_bot = vertice_coords
        elif self.point_mode == "Septum": context.scene.pos_septum = vertice_coords
        else:
            cons_print('Unsupported point mode input. Only Top, Bot and Septum available.')
            return{'CANCELLED'}    
        return{'FINISHED'}

class MESH_OT_ventricle_rotate(bpy.types.Operator):
    """Rotate Ventricle using the node-coordinates of the basal, apical and septum node"""
    bl_idname = 'heart.ventricle_rotate'
    bl_label = 'Rotate Ventricle using the node-coordinates of the basal, apical and septum node.' 
    def execute(self, context): 
        if not rotate_ventricle(context): return{'CANCELLED'}
        scene = context.scene
        view_layer = context.view_layer

        # --- Remember original selection & active object, also which meshes were selected ---
        original_selection = list(context.selected_objects)
        original_active = view_layer.objects.active
        selected_meshes_for_stl = [obj for obj in original_selection if obj.type == 'MESH']

        def restore_selection():
            bpy.ops.object.select_all(action='DESELECT')

            # Reselect original objects (only if they still exist)
            for obj in original_selection:
                if obj and obj.name in view_layer.objects:
                    view_layer.objects[obj.name].select_set(True)

            # Restore active object if it still exists
            if original_active and original_active.name in view_layer.objects:
                view_layer.objects.active = view_layer.objects[original_active.name]

        # ------------------------------------------------------------------
        # 1) Find ventricles only among the currently selected objects
        # ------------------------------------------------------------------
        ventricles = find_ventricle_objects(original_selection)
        if not ventricles:
            cons_print("Export ventricle: no ventricle_* meshes found in the current selection.")
            return {'CANCELLED'}

        # ------------------------------------------------------------------
        # 2) Node-connectivity check using only these ventricles
        # ------------------------------------------------------------------
        bpy.ops.object.select_all(action='DESELECT')
        for obj in ventricles:
            obj.select_set(True)
        view_layer.objects.active = ventricles[0]

        # This prints success message itself if everything is fine
        if not check_node_connectivity(context):
            restore_selection()
            return {'CANCELLED'}

        # ------------------------------------------------------------------
        # 3) Determine export directories:
        #    uses whatever is at the import directory as the export directory
        # ------------------------------------------------------------------
        export_dir_raw = (scene.ventricle_import_dir or "").strip()
        if not export_dir_raw:
            export_dir_raw = "//"

        base_dir = os.path.normpath(bpy.path.abspath(os.path.join(export_dir_raw,"rotated")))
        connectivity_dir = os.path.normpath(os.path.join(base_dir, "Connectivity"))

        try:
            os.makedirs(connectivity_dir, exist_ok=True)
        except Exception as e:
            cons_print(f"Export ventricle: cannot create directory '{connectivity_dir}': {e}")
            restore_selection()
            return {'CANCELLED'}

        # ------------------------------------------------------------------
        # 4) Export connectivity (faces) from the first ventricle
        # ------------------------------------------------------------------
        ref_obj = ventricles[0]
        mesh = ref_obj.data

        # Ensure mesh is triangulated (for connectivity consistency)
        for poly in mesh.polygons:
            if len(poly.vertices) != 3:
                cons_print(
                    f"Export ventricle: non-triangular face {poly.index} in "
                    f"'{ref_obj.name}'. Please triangulate the mesh first."
                )
                restore_selection()
                return {'CANCELLED'}

        faces_path = os.path.join(connectivity_dir, "ventricle_faces.txt")
        try:
            with open(faces_path, "w") as f:
                for poly in mesh.polygons:
                    v0, v1, v2 = poly.vertices
                    f.write(f"{v0} {v1} {v2}\n")
        except Exception as e:
            cons_print(f"Export ventricle: error writing faces file: {e}")
            restore_selection()
            return {'CANCELLED'}

        # ------------------------------------------------------------------
        # 5) Export per-frame vertex coordinates for ventricles
        #    ALWAYS named ventricle_verts_0, ventricle_verts_1, ...
        #    Also collect data for the manifest.
        # ------------------------------------------------------------------
        manifest_entries = []  # (index, stl_file, verts_file, source_name)

        for export_index, obj in enumerate(ventricles):
            verts_filename = f"ventricle_verts_{export_index}.txt"
            verts_path = os.path.join(connectivity_dir, verts_filename)

            try:
                with open(verts_path, "w") as f:
                    for v in obj.data.vertices:
                        co_world = obj.matrix_world @ v.co
                        f.write(f"{co_world.x:.8f} {co_world.y:.8f} {co_world.z:.8f}\n")
            except Exception as e:
                cons_print(
                    f"Export ventricle: error writing vertices for '{obj.name}': {e}"
                )
                restore_selection()
                return {'CANCELLED'}

            stl_filename = f"ventricle_{export_index}.stl"
            manifest_entries.append(
                (export_index, stl_filename, os.path.join("Connectivity", verts_filename), obj.name)
            )

        # ------------------------------------------------------------------
        # 6) Restore original selection (for user convenience)
        # ------------------------------------------------------------------
        restore_selection()

        # ------------------------------------------------------------------
        # 7) Export STLs
        #       - ventricles -> ventricle_0.stl, ventricle_1.stl, ...
        #       - all other originally selected meshes -> <safe_name>.stl
        # ------------------------------------------------------------------
        depsgraph = context.evaluated_depsgraph_get()

        # 7a) ventricles with canonical names
        for export_index, obj in enumerate(ventricles):
            stl_path = os.path.join(base_dir, f"ventricle_{export_index}.stl")
            export_object_to_stl(obj, stl_path, depsgraph=depsgraph)

        # 7b) other selected meshes (non-ventricles) with their own names
        other_meshes = [obj for obj in selected_meshes_for_stl if obj not in ventricles]
        for obj in other_meshes:
            safe_name = re.sub(r'[^0-9A-Za-z_]+', '_', obj.name)
            stl_path = os.path.join(base_dir, f"{safe_name}.stl")
            export_object_to_stl(obj, stl_path, depsgraph=depsgraph)

        # ------------------------------------------------------------------
        # 8) Write manifest file mapping ventricle indices to files + original names
        # ------------------------------------------------------------------
        manifest_path = os.path.join(base_dir, "ventricle_export_manifest.txt")
        try:
            with open(manifest_path, "w") as mf:
                mf.write("# Ventricle export manifest\n")
                mf.write("# base directory for STL and this manifest: {}\n".format(base_dir))
                mf.write("# Connectivity files are in subfolder: {}\n".format(connectivity_dir))
                mf.write("#\n")
                mf.write("# index  STL_file            verts_file                         source_object_name\n")
                for idx, stl_file, verts_file, source_name in manifest_entries:
                    mf.write(
                        f"{idx:3d}  {stl_file:18s}  {verts_file:30s}  {source_name}\n"
                    )
        except Exception as e:
            cons_print(f"Export ventricle: error writing manifest file: {e}")
            # Don't cancel export; STLs and verts already written.

        # Single, light console summary line
        cons_print(
            f"Export ventricle: STL -> '{base_dir}', connectivity -> '{connectivity_dir}', manifest -> '{manifest_path}'"
        )

        return {'FINISHED'}

def rotate_ventricle(context):
    """Rotate ventricle geometry using three points on the ventricle"""
    ## Coniditions to terminate the code.
    if bpy.context.mode != 'OBJECT': bpy.ops.object.editmode_toggle() # Toggle to object mode.
    if len(bpy.context.selected_objects) < 1:# Only works if and object is selected
        cons_print("No object selected.")
        return False
    ## Precompute the rotation angles using the relative positions between top, bottom and septum node.
    # Initialize points.
    top = mathutils.Vector((context.scene.pos_top[0], context.scene.pos_top[1], context.scene.pos_top[2]))
    bottom = mathutils.Vector((context.scene.pos_bot[0], context.scene.pos_bot[1], context.scene.pos_bot[2]))
    septum = mathutils.Vector((context.scene.pos_septum[0], context.scene.pos_septum[1], context.scene.pos_septum[2]))
    vec_difference = np.array([top.x - bottom.x, top.y - bottom.y, top.z - bottom.z]) # Compute difference between Top and Bottom to compute the angle for the first rotation.
    # First rotation precomputation - X.
    angle_x = get_rotation_angle(vec_difference[1], vec_difference[2]) # Get angle.
    # Compute rotated top and septum node using rotation matrix around x-axis.
    rot_matrix = np.array([[1, 0, 0], [0, math.cos(angle_x), -math.sin(angle_x)], [0, math.sin(angle_x), math.cos(angle_x)]]) 
    rot_top = rot_matrix.dot(vec_difference) 
    rot_septum = rot_matrix.dot(septum-bottom) 
    # Second rotation precomputation - Y.
    angle_y = - get_rotation_angle(rot_top[0], rot_top[2]) # Get angle. Turn the rotation direction with a minus sign.
    # Compute second rotation of septum node using rotation matrix around y-axis.
    rot_matrix_two = np.array([[math.cos(angle_y), 0, math.sin(angle_y)], [0, 1, 0], [-math.sin(angle_y), 0, math.cos(angle_y)]]) 
    double_rot_top = rot_matrix_two.dot(rot_top)
    double_rot_septum = rot_matrix_two.dot(rot_septum)
    context.scene.pos_top = (round(abs(double_rot_top[0]), 6), round(abs(double_rot_top[1]), 6), round(abs(double_rot_top[2]), 6)) # Update UI top-variables.
    # Third rotation precomputation - Z.
    angle_z = get_rotation_angle(double_rot_septum[0], double_rot_septum[1]) # Get angle.
    # Compute third rotation of septum node using rotation matrix around y-axis.
    rot_matrix_three = np.array([[math.cos(angle_z), -math.sin(angle_z), 0], [math.sin(angle_z), math.cos(angle_z), 0], [0, 0, 1]]) 
    third_rot_septum = rot_matrix_three.dot(double_rot_septum)
    context.scene.pos_septum = (round(abs(third_rot_septum[0]), 6), round(abs(third_rot_septum[1]), 6), round(abs(third_rot_septum[2]), 6)) # Update UI septum-variables.
    ## Translation and rotation-process for all selected objects.
    # Translation.
    if context.scene.pos_bot != (0, 0, 0):   
        # Translate Coordinate system to (0,0,0) by subtracting the bottom node for easier rotation
        bpy.ops.transform.translate(value=(-bottom), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.object.transform_apply(location=True, scale=False, rotation=False)
        # Update UI bottom-variables.
        context.scene.pos_bot = (0, 0, 0)
    # Rotation operations around x-, y- and z-axis. 
    bpy.ops.transform.rotate(value=angle_x, orient_axis='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
    bpy.ops.transform.rotate(value=angle_y, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
    bpy.ops.transform.rotate(value=angle_z, orient_axis='Z', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
    # Apply all transformations.
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return True

def get_rotation_angle(numerator, denominator):
    """Function to quickly and reliably compute the rotation angle. Python tan-functions did not give the correct angles for some cases"""
    angle = 0
    if denominator > 0: angle = math.atan(numerator/denominator)
    elif denominator < 0:
        if numerator > 0: angle = math.pi + math.atan(numerator/denominator)
        elif numerator < 0: angle = - math.pi + math.atan(numerator/denominator)
    else:
        if numerator > 0: angle = math.pi    
        elif numerator < 0: angle = - math.pi 
    return angle

def get_neighbour_vertices(v):
    """Return neighbouring vertices of a vertex"""
    neighbours_index = []
    for e in v.link_edges: neighbours_index.append(e.other_vert(v).index)
    return neighbours_index
    
class MESH_OT_remove_basal(bpy.types.Operator): 
    """Remove the basal region using a threshold value"""
    bl_idname = 'heart.remove_basal'
    bl_label = 'Remove basal region.'
    def execute(self, context): 
        remove_multiple_basal_region(context)
        return{'FINISHED'} 

def remove_multiple_basal_region(context):
    """Remove multiple basal regions selecting the EDV as the reference element"""
    cons_print(f"Removing original basal region.")
    # Setup objects before basal region removal.
    if not context.selected_objects:
        cons_print("No elements selected.")
        return False
    selected_objects = context.selected_objects
    # Find reference object and create a copy of it as a reference object. This is either the mean or max value between all selected objects.
    if bpy.types.Scene.mean_reference: reference_copy = copy_object(find_reference_ventricle_mean(selected_objects), 'reference')  
    else: reference_copy = copy_object(find_reference_ventricle_max(selected_objects), 'reference')
    cons_print(f"Chosen reference object: {bpy.types.Scene.reference_object_name}")
    for obj in selected_objects: obj.select_set(False) # Deselect objects.
    # Basal region removal. First for reference, then remaining objects.
    deleted_verts = remove_basal_region(context, reference_copy, []) # Remove in reference object
    for obj in selected_objects: remove_basal_region(context, obj, deleted_verts) 
    # Longitudinal shift of each ventricle to match reference object, reducing volume discrepancy between systole and diastole between raw data and reconstructed data.
    shift_ventricles_longitudinally(context, selected_objects)
    context.scene.ref_maxima, context.scene.ref_minima = get_min_max(reference_copy)    
    # Cleanup.
    for obj in selected_objects: obj.select_set(True) # Reselect objects from original selection after main operations are executed.
    bpy.data.objects.remove(bpy.data.objects["reference"], do_unlink=True) # Remove reference object.

def remove_basal_region(context, obj, del_nodes):
    """Remove basal region of the ventricle using a threshold"""
    if obj.mode == 'OBJECT': 
        bpy.ops.object.mode_set(mode='EDIT') # Toggle to object mode.
    bpy.context.tool_settings.mesh_select_mode = (True, False, False) # Go into vertex mode
    bpy.ops.object.mode_set(mode='OBJECT') # Toggle to object mode.
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    deselect_object_vertices(obj)
    # Select vertices to delete (above z-coordinate threshold).
    bm = transfer_data_to_mesh(obj) 
    if del_nodes == []: # Initialisation of deleted node selection for reference object.
        for v in bm.verts:
            vertice_coords = obj.matrix_world @ v.co # Transfer to global coordinate system.
            if vertice_coords[2] > context.scene.remove_basal_threshold: # Only vertices above threshold are selected to be deleted.
                v.select = True
                del_nodes.append(v.index)
    else:
        for v in bm.verts:
            if v.index in del_nodes: v.select = True
    bm.to_mesh(obj.data) # Update selection to object.
    ## Remove selected nodes.
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.delete_edgeloop()
    bpy.ops.object.mode_set(mode='OBJECT')
    # Save upper apical edge loop in a vertex group to be selected again later on.
    bm = transfer_data_to_mesh(obj)
    marked_verts = [v.index for v in bm.verts if v.select]
    vg_upper_apical = "upper_apical_edge_loop"
    vg_orifice = obj.vertex_groups.new(name = vg_upper_apical)
    vg_orifice.add(marked_verts, 1, 'ADD' )
    # Remove remaining face create after delete_edgeloop().
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.delete(type='FACE') 
    # Refinement
    refine_upper_apical_edge_loop(vg_orifice)
    smooth_apical_region(obj, vg_orifice)
    # Close function.
    obj.select_set(False)
    return del_nodes

def refine_upper_apical_edge_loop(vg_orifice):
    """Flatten the upper apical edge loop"""
    # Select upper apical edge loop with vertex group.
    bpy.ops.object.vertex_group_set_active(group=vg_orifice.name)
    bpy.ops.object.vertex_group_select()
    # Smooth highest edge loop of apical region aligning the vertices onto a plane.
    change_to_viewpoint('TOP') # Change view to top view.
    bpy.ops.mesh.looptools_relax(input='selected', interpolation='linear', iterations='5', regular=True) # Reduce spikes on the highest edge loop.
    bpy.ops.mesh.looptools_flatten(influence=90, lock_x=True, lock_y=True, lock_z=False, plane='view', restriction='none')

def change_to_viewpoint(alignment_type):
    """Update 3D view"""
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            ctx = {
                "window": bpy.context.window, # current window, could also copy context
                "area": area, # our 3D View (the first found only actually)
                "region": None # just to suppress PyContext warning, doesn't seem to have any effect
            }
            bpy.ops.view3d.view_axis(ctx, type=alignment_type, align_active=False)
            area.spaces.active.region_3d.update()  # <---

def subdivide_last_edge_loop(obj, vg_orifice):
    """Subdivide last edge loop in two steps before bridging for a better transition between coarse apical and fine basal mesh"""
    # Select upper apical edge loop with vertex group.
    deselect_object_vertices(obj)
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.object.vertex_group_set_active(group=vg_orifice.name)
    bpy.ops.object.vertex_group_select()
    # Iteratively subdivide edge loop with surrounding mesh creating a less skewed transition between subdivides and unsibdivided elements.
    bpy.ops.mesh.select_more()
    bpy.ops.mesh.subdivide(number_cuts=1, ngon=False)
    bpy.ops.mesh.select_less()
    bpy.ops.mesh.select_less()
    bpy.ops.object.mode_set(mode='OBJECT')
    # Re-initialize vertex group for upper apical edge loop.
    bpy.ops.object.mode_set(mode='OBJECT')
    bm = transfer_data_to_mesh(obj)
    selected_verts = [v.index for v in bm.verts if v.select] # Re-read selected vertices.
    vg_name = vg_orifice.name
    if vg_orifice is not None: obj.vertex_groups.remove(vg_orifice) # Delete previous vertex group to make room for new one.
    vg_orifice = obj.vertex_groups.new(name = vg_name)
    vg_orifice.add(selected_verts, 1, 'ADD' )
    return vg_orifice

def smooth_apical_region(obj, vg_orifice):
    """Smooth apical region in the region of the cut"""
    deselect_object_vertices(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.vertex_group_set_active(group=vg_orifice.name)
    # Find vertices to select initially for smoothing.
    init_edge_loops = 2 # Initially selected vertices for smoothing operation.
    bpy.ops.object.vertex_group_select()
    for i in range(init_edge_loops): 
        bpy.ops.mesh.select_more()
    bpy.ops.object.vertex_group_deselect()
    # # Continuously weaker smoothing operations with more nodes to create a smooth transition between smoothed and unsmoothed region.
    n_smooth_iter = 3
    for i in range(n_smooth_iter): 
        bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=n_smooth_iter+1-i)
        bpy.ops.mesh.select_more()
        bpy.ops.object.vertex_group_deselect()
    bpy.ops.object.mode_set(mode='OBJECT')

def shift_ventricles_longitudinally(context, objects):
    """Shift ventricle to reference ventricle"""
    for obj in objects:
        max_obj_val, min_obj_val = get_min_max(obj)
        shift_distance =  context.scene.remove_basal_threshold - max_obj_val[2] 
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        obj['long_shift'] = shift_distance
        bpy.context.object.location[2] = shift_distance
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.select_set(False)

class MESH_OT_build_valves(bpy.types.Operator):
    """Create geometry for mitral and aortic valve"""
    bl_idname = 'heart.build_valve'
    bl_label = 'Create geometry for mitral or aortic valve.'
    def execute(self, context):
        ratio_annulli = 1
        for obj in context.selected_objects:
            if not build_both_valves(context, obj, ratio_annulli): return{'CANCELLED'} 
            merge_overlap(threshold = 0.0001) # Protection against double insertion of valves.
        return{'FINISHED'}

def build_both_valves(context, obj, ratio):
    """Build both the aortic and mitral valve"""
    aortic_min = build_valve(context, obj, valve_mode = "Aortic" , ratio = ratio)
    mitral_min = build_valve(context, obj,  valve_mode = "Mitral", ratio =  ratio)
    return aortic_min, mitral_min

def build_valve(context, obj,  valve_mode, ratio):
    """Build ventricle valve and connect it to current geometry"""
    if valve_mode == "Aortic": obj_name = f"A{context.scene.approach}_AV_Boundary"
    elif valve_mode == "Mitral": obj_name = f"A{context.scene.approach}_MV_Boundary"
    else: return False
    return add_and_join_object(context, obj, obj_name, valve_mode, ratio)

def add_and_join_object(context, obj, new_obj_name, valve_mode, ratio):
    """Add an object and join it with the currently selected geometries"""
    # Create new object by copying it from existing object.
    new_obj = copy_object(new_obj_name, "new_obj")
    new_obj.select_set(True)
    bpy.context.view_layer.objects.active = new_obj
    # Modify the copied object to fit in the correct place.
    scale_rotate_translate_object(context, new_obj, valve_mode, ratio)
    maxim, minim = get_min_max(new_obj)
    # Combine new object with selected object.
    join_objects(obj, new_obj) 
    return minim[2]

def scale_rotate_translate_object(context, new_obj, valve_mode, ratio):
    """Scale, rotate and shift object and save transformation"""
    if valve_mode == "Mitral" or valve_mode == "Aortic":
        translation, angles, radius_vertical, radius_horizontal = get_valve_data(context, valve_mode) # Get scale from UI-data.
    else:
        # For support structure only. For now no changes to rotation, but slightly increased scaling.
        angles = (0,0,0)
        radius_vertical = 2
        radius_horizontal = 2
        # Find position furthest away from any other points before Poisson surface reconstruction to reduce the size of the largest wholes in the geometry. This will improve Poisson surface reconstruction.
        x_center = (context.scene.ref_minima[0] + context.scene.ref_maxima[0]) / 2
        y_value = (context.scene.translation_aortic[1] + context.scene.translation_mitral[1]) / 2
        z_value = (context.scene.translation_aortic[2] + context.scene.translation_mitral[2]) / 2
        if valve_mode == "Front":
            x_value = (x_center + context.scene.ref_maxima[0]) / 2
            translation = (x_value, y_value, z_value)
        elif valve_mode =="Back":
            x_value = (x_center +  context.scene.ref_minima[0]) / 2
            translation = (x_value, y_value, z_value)
        else:
            return False
    new_obj.scale = (ratio * radius_vertical, ratio * radius_horizontal, ratio * (radius_horizontal + radius_vertical) / 2) # Scale object with given ratio.
    new_obj.rotation_euler = angles # Rotate object by input angles.
    new_obj.location = translation # Translate object with input translation.
    if new_obj.mode == 'OBJECT': bpy.ops.object.mode_set()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True) # Apply changes.
    bpy.ops.object.mode_set()

def get_min_max(obj):
    """Return smallest and highest value of an object in each dimension"""
    maxim = np.array([-1000.0, -1000.0, -1000.0])
    minim = np.array([1000.0, 1000.0, 1000.0])
    for p in obj.data.vertices: # Find maxima and minima.
        vertice_coords = obj.matrix_world @ p.co # Transform to global coordinate system.
        if vertice_coords[0] > maxim[0]: maxim[0] = vertice_coords[0]
        if vertice_coords[0] < minim[0]: minim[0] = vertice_coords[0]
        if vertice_coords[1] > maxim[1]: maxim[1] = vertice_coords[1]
        if vertice_coords[1] < minim[1]: minim[1] = vertice_coords[1]
        if vertice_coords[2] > maxim[2]: maxim[2] = vertice_coords[2]
        if vertice_coords[2] < minim[2]: minim[2] = vertice_coords[2]
    return maxim, minim

def merge_overlap(threshold):
    """Merge overlapping vertices of currently selected object"""
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=threshold, use_sharp_edge_from_normals=False, use_unselected=False)
    bpy.ops.object.mode_set(mode='OBJECT') 

class MESH_OT_support_struct(bpy.types.Operator):
    """Create supporting structure to help poisson algorithm accurately build basal ventricle region"""
    bl_idname = 'heart.support_struct'
    bl_label = 'Create supporting structure to help poisson algorithm accurately build basal ventricle region.'
    def execute(self, context):
        for obj in context.selected_objects:
            build_support_structure(context, obj, ratio=1.1)
            merge_overlap(threshold = 0.0001) # Protection against double insertion of valves.
        return{'FINISHED'}

def build_support_structure(context, obj, ratio):
    """Build support structure to help the poisson surface reconstrucion algorithm create a smooth surface after Poisson surface reconstruction"""
    aortic_min_up, mitral_min_up = build_both_valves(context, obj, ratio) # Larger annulus structure (upscaled).
    aortic_min_down, mitral_min_down =  build_both_valves(context, obj, 1 / ratio) # Smaller annulus structure (downscaled).
    build_both_centers(context, obj)
    build_ouside_support_structure(context, obj)
    return aortic_min_up, mitral_min_up, aortic_min_down, mitral_min_down

def build_both_centers(context, obj):
    """Build both the aortic and mitral valve"""
    if build_center(context, obj, valve_mode = "Aortic" ) and build_center(context, obj,  valve_mode = "Mitral"): return True
    else: return False

def build_center(context, obj,  valve_mode):
    """Build ventricle valve and connect it to current geometry"""
    if valve_mode == "Aortic": obj_name = f"A{context.scene.approach}_AV_Center"
    elif valve_mode == "Mitral": obj_name = f"A{context.scene.approach}_MV_Center"
    else: return False
    return add_and_join_object(context, obj, obj_name, valve_mode, 1)

def build_ouside_support_structure(context, obj):
    """Build support structure in large holes between valves and the apical region"""
    add_and_join_object(context, obj, f"A{context.scene.approach}_AV_Center", "Front", 1)
    add_and_join_object(context, obj, f"A{context.scene.approach}_AV_Center", "Back", 1)

class MESH_OT_poisson(bpy.types.Operator):
    """Apply Poisson surface reconstrucion to point cloud creating a surface mesh"""
    bl_idname = 'heart.poisson'
    bl_label = 'Apply Poisson surface reconstrucion to point cloud creating a surface mesh.'
    def execute(self, context):
        if len(bpy.context.selected_objects) == 0 or bpy.context.mode != 'OBJECT': return{'CANCELLED'} # Check if any object is selected and correct context mode.
        for object in bpy.context.selected_objects: create_poisson_from_object_pointcloud(context, object) # Repeat Poisson-surface-reconstruction algorithm for all selected objects.
        return{'FINISHED'}

def create_poisson_from_object_pointcloud(context, obj): 
    """Create poisson surface reconstruction for a single point cloud"""
    point_data = np.asarray(obj.data.vertices) # Get point data of current object.
    # Initialize and fill entries of object vertices.
    object_vertices = np.empty(shape=[0, 3])
    for point in point_data:
        vertice_coords = obj.matrix_world @ point.co # Rotate points to put them in global coordinate system.
        # Append new vertex to point cloud array.
        new_point = np.array([[vertice_coords[0], vertice_coords[1], vertice_coords[2]]])
        object_vertices = np.concatenate((object_vertices, new_point), axis=0)
    # Create Point cloud object and fill it with points.
    point_cloud_data = o3d.geometry.PointCloud()
    point_cloud_data.points = o3d.utility.Vector3dVector(object_vertices)
    # Prepare point cloud for poisson surface reconstruction. It needs the normals of the points in the point cloud.
    point_cloud_data.estimate_normals()
    point_cloud_data.normalize_normals()
    point_cloud_data.orient_normals_consistent_tangent_plane(20)
    # Apply poisson surface reconstruction.
    poisson_mesh, poisson_dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud_data, depth = context.scene.poisson_depth, width=0, scale=1.1, linear_fit=False, n_threads= 1)
    # Initialize empty arrays for object data and assign faces vertices and edges.
    vertices, edges, faces = ([] for i in range(3))
    vertices = poisson_mesh.vertices
    faces = poisson_mesh.triangles
    # Create object from vertices, edges and faces in Blender.
    emptyMesh = bpy.data.meshes.new('emptyMesh')
    emptyMesh.from_pydata(vertices, edges, faces)
    emptyMesh.update()
    poisson_obj = bpy.data.objects.new(obj.name + "_poisson", emptyMesh)    
    bpy.context.collection.objects.link(poisson_obj)
    # Change selection.
    obj.select_set(False)
    poisson_obj.select_set(True)
    bpy.context.view_layer.objects.active = poisson_obj
    obj.hide_set(True)
    return poisson_obj

class MESH_OT_create_valve_orifice(bpy.types.Operator):
    """Remove vertices blocking the valve entries into a single face, delete this face and create a vertex group for all vertices around that face"""
    bl_idname = 'heart.create_valve_orifice'
    bl_label = 'Remove vertices blocking the valve entries into a single face, delete this face and create a vertex group for all vertices around that face.'
    def execute(self, context):
        # Check for the right amount of objects and context mode.
        if len(bpy.context.selected_objects) == 0 or bpy.context.mode != 'OBJECT': return{'CANCELLED'}
        for obj in context.selected_objects:
            deselect_object_vertices(obj) # Deselect all vertices as preparation for the deletion of the valve orifice.
            # Delete nodes in areas around the valves to create valve orifices.
            create_valve_orifice(context, "Aortic")
            create_valve_orifice(context, "Mitral")
        return{'FINISHED'}

def create_valve_orifice(context, valve_mode): 
    """Create orifice in the geometry, where valve interface nodes will be inserted later"""
    # Dissolve vertices inside of a given area-object placed around the valve vertices.
    select_valve_vertices(context, valve_mode) 
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.dissolve_verts()
    bpy.ops.object.mode_set(mode='OBJECT')
    # Remove face remaining after dissolving vertices.
    translation, angles, radius_vertical, radius_horizontal = get_valve_data(context, valve_mode)
    obj = bpy.context.active_object
    if obj.mode == 'EDIT': bpy.ops.object.mode_set() # Toggle to object mode.
    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    bm = transfer_data_to_mesh(obj)
    vertices_orifice = []
    # Select vertices of newly created face after dissolving.
    for f in bm.faces:
        f.select = False
        # Compute difference between center and translation vector and check if it is smaller than the smaller valve radius.
        if distance_vec(f.calc_center_median(), translation) < min(radius_vertical, radius_horizontal)  / 2: 
            for v in f.verts: vertices_orifice.append(v.index)
            f.select = True
    # Delete face in orifice.
    faces = [f for f in bm.faces if f.select]
    bmesh.ops.delete(bm, geom = faces, context = 'FACES_ONLY')
    bm.to_mesh(obj.data)
    # Create vertex group containing orifice edge loop vertices.
    vg_orifice = obj.vertex_groups.new( name = f"{valve_mode}_orifice")
    vg_orifice.add( vertices_orifice, 1, 'ADD')
    # Remove troubling vertices(vertices with 2 neighbours) in (currently selected) orifice vertex group and smooth this edge loop.
    smooth_relax_edgeloop(obj, vg_orifice) 
    bpy.ops.object.mode_set(mode='OBJECT')
    return True

def select_valve_vertices(context, valve_mode):
    """Select all vertices of a given valve"""
    if 'Valve_area' in [obj.name for obj in bpy.data.objects]: bpy.data.objects.remove(bpy.data.objects['Valve_area'], do_unlink=True) # Remove valve area object, if an object with the same name already exists.
    original_obj = bpy.context.active_object
    if valve_mode == "Mitral":   valve_area = copy_object(f'A{context.scene.approach}_MV_Hull', 'Valve_area')
    elif valve_mode == "Aortic": valve_area = copy_object(f'A{context.scene.approach}_AV_Hull', 'Valve_area')
    else: return False
    # Rescale, translate and rotate valve area.
    scale_rotate_translate_object(context, valve_area, valve_mode, ratio=1.025) 
    # Select all vertices in the valve area object.
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action="DESELECT")
    mesh=bmesh.from_edit_mesh(bpy.context.object.data)
    cut_obj_matrix = valve_area.matrix_world.inverted()
    mat = cut_obj_matrix @ original_obj.matrix_world         
    selected_verts = [v
        for v in mesh.verts
        if is_inside((mat @ v.co), valve_area)]  
    for v in selected_verts: 
        if v.co.z > -1: v.select = True 
    # Exclude vertice in lower basal edge loopa
    for group in bpy.context.active_object.vertex_groups:
        if group.name == "lower_basal_edge_loop":
            bpy.ops.object.vertex_group_set_active(group=group.name)
            bpy.ops.object.vertex_group_deselect()
            break
    # Remove vertices
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.data.objects.remove(bpy.data.objects[valve_area.name], do_unlink=True) # Remove valve area object.
    
def distance_vec(point1, point2) -> float:
    """Calculate distance between two nodes"""
    return math.sqrt((point2[0] - point1[0]) ** 2 +
                     (point2[1] - point1[1]) ** 2 +
                     (point2[2] - point1[2]) ** 2)

def is_inside(p, cut_obj):
    """Check if a point p is inside the object cut_obj"""
    result, point, normal, face = cut_obj.closest_point_on_mesh(p)
    if not result: return False
    p2 = point-p
    v = p2.dot(normal)
    return not (v < 0.0)

def get_valve_data(context, valve_mode):
    """ Return data for specific valve type"""
    if valve_mode == "Aortic":
        translation = context.scene.translation_aortic
        angles = np.radians(context.scene.angle_aortic)
        radius_vertical = context.scene.aortic_radius
        radius_horizontal = context.scene.aortic_radius
    elif valve_mode == "Mitral":
        translation = context.scene.translation_mitral
        angles = np.radians(context.scene.angle_mitral)
        radius_vertical = context.scene.mitral_radius_long
        radius_horizontal = context.scene.mitral_radius_small
    elif valve_mode == "Front":
        pass
    elif valve_mode == "Back":
        pass

    else: return False, False, False, False
    return translation, angles, radius_vertical, radius_horizontal 
    
class MESH_OT_connect_valves(bpy.types.Operator):
    """Connect valve interface nodes with orifice for mitral and aortic valve"""
    bl_idname = 'heart.connect_valve'
    bl_label = 'Connect valve interface nodes with orifice for mitral and aortic valve.'
    def execute(self, context):
        connect_valve_orifice(context, "Aortic", valve_index = 4)
        connect_valve_orifice(context, "Mitral", valve_index = 4)
        return{'FINISHED'}

def connect_valve_orifice(context, valve_mode, valve_index):
    """Connect orifices around valves with surrounding mesh nodes"""
    obj = bpy.context.active_object
    build_valve_surface(context, obj, valve_mode = valve_mode, ratio = 1, valve_index = valve_index) # Create interface valve nodes.
    bpy.ops.object.mode_set(mode='EDIT') # Change selection mode in edit mode for brige loop operator.
    bpy.ops.mesh.bridge_edge_loops()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)

def smooth_relax_edgeloop(obj, vg_orifice):
    """Relax selected edge loop such that vertices with only two edges get deleted before relaxation, because these vertices will create skew triangles, when using the bridge_edgeloop() command"""
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.context.tool_settings.mesh_select_mode = (True, False, False) # Activate edge mode in edit mode.
    i = 0 # Breakup-condition for while-loop to prevent infinite loops.
    while select_vertices_outside_of_edge_loop(obj) and i < 100: # Select vertices with only two neighbours. 100 is to prevent infinite loops.
        i = i + 1
        bpy.ops.object.mode_set(mode='OBJECT') # Update Mesh in Blender.
        bpy.ops.object.mode_set(mode='EDIT') 
        bpy.ops.mesh.delete(type='VERT') # Remove selected vertices.
        # Reselect vertices in orifice vertex group.
        bpy.ops.object.vertex_group_set_active(group=str(vg_orifice.name))
        bpy.ops.object.vertex_group_select()
    # Select vertices in vertex group of orifice vertices.
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.object.vertex_group_set_active(group=str(vg_orifice.name))
    bpy.ops.object.vertex_group_select()
    bpy.ops.object.vertex_group_set_active(group=str('lower_basal_edge_loop'))
    bpy.ops.object.vertex_group_deselect()
    # Relax orifice loop to reduce spikes in transition to valve.
    bpy.ops.mesh.looptools_relax(input='selected', interpolation='linear', iterations='5', regular=True)
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)

def select_vertices_outside_of_edge_loop(obj):
    """Select vertices in an edge loop with two or less neighbours"""
    # Transfer data into edit mode.
    me = obj.data
    bpy.ops.object.mode_set(mode='EDIT') 
    bm = bmesh.from_edit_mesh(me)
    # Initialize return type
    must_remove = False # No vertices needs to be removed.
    # Select vertices in edge loop with only two connecting vertices.
    for v in bm.verts:
        if len(get_neighbour_vertices(v)) <= 2 and v.select: 
            if vertex_in_vertex_group(obj, v, 'lower_basal_edge_loop'):
                v.select = False
                continue   
            v.select = True
            must_remove = True
        else: v.select = False
    bpy.ops.object.vertex_group_set_active(group="lower_basal_edge_loop")
    bpy.ops.object.vertex_group_deselect()
    # Return to object mode and update the mesh to the obeject.
    bm.select_flush_mode()   
    me.update()
    bpy.ops.object.mode_set(mode='OBJECT') 
    return must_remove

def vertex_in_vertex_group(obj, v, vertex_group_name):
    """Check if a vertex v is in a given vertex groupof an object"""
    if "lower_basal_edge_loop" in obj.vertex_groups:
        for vg_v in [vert for vert in bpy.context.object.data.vertices if bpy.context.object.vertex_groups[vertex_group_name].index in [i.group for i in vert.groups]]:
            if vg_v.index == v.index: return True
    return False

def build_valve_surface(context, obj, valve_mode, ratio, valve_index):
    """Copy valve object(mitral/aortic), combine it with the active geometry and select only boundary vertices. Boundary_only allows to delete all vertices except the boundary edge loop on the ventricle side"""
    # Assign names for further operations:
    if valve_mode == "Aortic": 
        obj_name = f"A{context.scene.approach}_AV_Surface"
        vg_boundary = "AV_Boundary"   # Vertex group of the boundary of the valve depending on valve_mode (MV_Boundary/AV_Boundary):
        vg_orifice = "Aortic_orifice"
    elif valve_mode == "Mitral": 
        vg_boundary = "MV_Boundary"
        vg_orifice = "Mitral_orifice"
        if context.scene.approach == 3 or context.scene.approach == 4: obj_name = f"A{context.scene.approach}_MV_Surface" # A3 or A4.
        elif valve_index in range(5): # Interpolated mitral valve (A5).
            obj_name = f"A{context.scene.approach}_MV_{valve_index}"
        else: return False
    else: return False
    add_and_join_object(context, obj, obj_name, valve_mode, ratio) # Add valve object for the given valve type.
    # Select valve boundary and orifice vertices.
    deselect_object_vertices(obj)
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.object.vertex_group_set_active(group=str(vg_boundary))
    bpy.ops.object.vertex_group_select()
    bpy.ops.object.vertex_group_set_active(group=str(vg_orifice))
    bpy.ops.object.vertex_group_select()
    bpy.ops.object.mode_set(mode='OBJECT') 

class MESH_OT_create_basal(bpy.types.Operator):
    bl_idname = 'heart.create_basal'
    bl_label = 'Create basal region of ventricle using the position and angles of the heart valves.'
    def execute(self, context):
        if not context.selected_objects:
            cons_print("No elements selected.")
            return False
        selected_objects = context.selected_objects
        for obj in selected_objects:
            if not mesh_create_basal(context, [obj]): return{'CANCELLED'}
        return{'FINISHED'} 

def mesh_create_basal(context, selected_objects):
    """Create Basal Regions for each of the selected objects"""
    if len(selected_objects) == 1:
        bpy.types.Scene.reference_object_name = selected_objects[0].name
    else:
        cons_print(f"Currently does not work :(")
        return False
    #reference_copy = copy_object(bpy.types.Scene.reference_object_name, 'basal_region')
    reference_copy = copy_object(selected_objects[0].name, 'basal_region')
    # Deselect objects.
    for obj in selected_objects: obj.select_set(False)
    # Operations to create basal region of the ventricle.
    basal_regions = create_basal_region_for_object(context, reference_copy)
    if not basal_regions: 
        cons_print(f"Error during the creation of the basal regions.")
        return False # If an error ocurred during creation of basal region, dont continue.
    # Cleanup.
    # Reselect objects to state previous to this operation and deselect (and hide for performance) created objects.
    for obj in selected_objects: obj.select_set(True)
    for basal in basal_regions:  
        basal.select_set(False)
        basal.hide_set(True)
    # Remove old basal region objects.
    if context.scene.approach == 5: bpy.data.objects.remove(bpy.data.objects["basal_ref"], do_unlink=True)
    bpy.data.objects.remove(bpy.data.objects["basal_region"], do_unlink=True)
    bpy.data.objects.remove(bpy.data.objects["basal_region_poisson"], do_unlink=True)
    return basal_regions

def find_reference_ventricle_max(objects): 
    """Find reference object with max volume and return its name"""
    max = 0
    for obj in objects:  
        bm = transfer_data_to_mesh(obj)
        volume = bm.calc_volume(signed=True)# Compute volume and append it to the volume list.
        if volume > max: # Find ventricle with maximum volume.
            max = volume
            bpy.types.Scene.reference_object_name = obj.name
    return bpy.types.Scene.reference_object_name

def find_reference_ventricle_mean(objects): 
    """Find reference object with max volume and return its name"""
    volumes = []
    for obj in objects: 
        bm = transfer_data_to_mesh(obj)
        volumes.append(bm.calc_volume(signed=True))
    mean_volume = sum(volumes) / len(volumes)
    diff_to_mean = 2 * max(volumes)
    for counter, obj in enumerate(objects):  
        if abs(volumes[counter] - mean_volume) < diff_to_mean: 
            diff_to_mean = abs(volumes[counter] - mean_volume)
            bpy.types.Scene.reference_object_name = obj.name
    return bpy.types.Scene.reference_object_name

def create_basal_region_for_object(context, reference_copy):
    """Create basal part for a given ventricle"""
    for obj in context.selected_objects: obj.select_set(False) # Deselect all objects.
    # Remove basal region of reference object.
    bpy.context.view_layer.objects.active = reference_copy
    deselect_object_vertices(reference_copy)
    # Add valve and support structure boundary nodes.
    aortic_min, mitral_min = build_both_valves(context, reference_copy, ratio= 1) # Valves have a annuli ratio of 1.
    aortic_min_up, mitral_min_up, aortic_min_down, mitral_min_down = build_support_structure(context, reference_copy, ratio=1.1) # Support structure for the valves have a annulie ratio of 1.1 and 1/1.1.
    # Compute height plane   
    context.scene.min_valves = np.amin([aortic_min, mitral_min, aortic_min_up, mitral_min_up, aortic_min_down, mitral_min_down]) # Compute minimal valve value for the position of the cutting plane.
    if not compute_height_plane(context): return False # Compute height plane for the removal of the apical region.
    # Poisson with remeshing and triangulation.
    poisson_basal = create_poisson_from_object_pointcloud(context, reference_copy)
    # Compute voxel_size for remesh and merge dependent of the smallest valve size. Reduced for approach 5.
    if context.scene.approach == 5: voxel_size = min(context.scene.aortic_radius, context.scene.mitral_radius_long, context.scene.mitral_radius_small) / 6 
    else: voxel_size = min(context.scene.aortic_radius, context.scene.mitral_radius_long, context.scene.mitral_radius_small) / 4 
    apply_voxel_remesh(voxel_size) # Apply Remesh for better mesh quality (remove small mesh elements with high cell skewness).
    # Triangulate remesh.
    bpy.ops.object.modifier_add(type='TRIANGULATE')
    bpy.ops.object.modifier_apply(modifier="Triangulate")
    merge_vertices(voxel_size) # Merge vertices close to each other.
    # Remove apical regions.
    remove_apical_region(context, poisson_basal)
    # Remove nodes in valve areas, insert of interface nodes and smooth the basal region.
    create_valve_orifice(context, "Aortic")
    create_valve_orifice(context, "Mitral")
    basal_regions = insert_valves_into_basal(context, poisson_basal) # Create exact inputs for the valve boundaries and connect it with the remaining basal region.
    for basal in basal_regions: basal.select_set(False)
    for basal in basal_regions: smooth_basal_region(context, basal, voxel_size) # Smooth basal region nodes excluding valves and lower edge loop.
    return basal_regions

def compute_height_plane(context): 
    """Compute height of the plane used to cut off the apical region from the basal region of the reference geometry"""
    if context.scene.min_valves <= context.scene.remove_basal_threshold: # The apical region extends over the basal region.
        cons_print(f"Error: Valves ({context.scene.min_valves}) lie beneath the highest point of the ventricle({context.scene.remove_basal_threshold}). Try a different setup for valve position or dissolve loops.")
        return False
    elif context.scene.min_valves < 1.025 * context.scene.remove_basal_threshold: # Basal and apical region lie very close to one another. This could lead to large kinks in the geometry.
        cons_print("Info: The basal and apical region are very close to one another. The geometry may contain large kinks especially in the connection between those regions. Try a higher dissolve loop number or higher z-value for the input valves.")
    else: pass # Basal and apical region have enough distance.
    context.scene.height_plane = (context.scene.remove_basal_threshold + context.scene.min_valves) / 2 # Choose z-value between lowest valve vertex and highest basal vertex.
    return True

def apply_voxel_remesh(voxel_size):
    """Apply remesh modifier with given depth to currently active object"""
    bpy.ops.object.modifier_add(type='REMESH')
    bpy.context.object.modifiers["Remesh"].mode = 'VOXEL'
    bpy.context.object.modifiers["Remesh"].voxel_size = voxel_size
    bpy.context.object.modifiers["Remesh"].adaptivity = 0
    bpy.context.object.modifiers["Remesh"].use_smooth_shade = False
    bpy.ops.object.modifier_apply(modifier="Remesh")

def merge_vertices(voxel_size):
    """Merge all vertices dependent on voxel size to eliminate vertices close to each other and thus reducing skewness in certain areas"""
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=voxel_size / 2, use_sharp_edge_from_normals=False, use_unselected=True)
    bpy.ops.object.mode_set(mode='OBJECT')

def remove_apical_region(context, obj):
    """Remove the apical ventricle region from the geometry to create solely the basal region used for all timeframes"""
    # Find vertices to delete.
    deselect_object_vertices(obj)
    bm = transfer_data_to_mesh(obj)
    for v in bm.verts:
        vertice_coords = obj.matrix_world @ v.co # Transfer to global coordinates.
        if vertice_coords[2] < context.scene.height_plane: v.select = True # Only vertices below threshold (height-plane) shall be deleted.
        else:  v.select = False
    bm.to_mesh(obj.data) # Transfer selection to object. 
    # Remove selected vertices.
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.delete_edgeloop()
    bpy.ops.object.mode_set(mode='OBJECT')
    # Save bottom edge loop to be selected again later on.
    bm = transfer_data_to_mesh(obj)
    marked_verts = [v.index for v in bm.verts if v.select] # Saved selected vertices.
    # Create vertex group for lower basal edge loop and add selected vertices to it.
    vg_lower_basal = "lower_basal_edge_loop"
    vg_orifice = obj.vertex_groups.new( name = vg_lower_basal)
    vg_orifice.add(marked_verts, 1, 'ADD' )
    # Remove remaining face created during delete_edgeloop() to create the orifice.
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.delete(type='FACE') 
    # Select vertex group.
    bpy.ops.object.vertex_group_set_active(group=str(vg_lower_basal))
    bpy.ops.object.vertex_group_select()
    # Relax and flatten lower edge loop.   
    bpy.ops.mesh.looptools_relax(input='selected', interpolation='linear', iterations='1', regular=True) # Reduce spikes on the cutting edge loop.
    bpy.ops.mesh.looptools_flatten(influence=100, lock_x=False, lock_y=False, lock_z=False, plane='best_fit', restriction='none')

def insert_valves_into_basal(context, poisson_basal): 
    """Insert valve geometry into geometry and connect it to orifice"""
    connect_valve_orifice(context, "Aortic", 0) # Insert aortic valve disk into basal region. 
    poisson_basal.select_set(False)
    basal_regions = []
    # Porous mitral valve.
    if context.scene.approach == 3 or context.scene.approach == 4: 
        # Copy basal region.
        bpy.ops.object.mode_set(mode='OBJECT')
        curr_basal = copy_object(poisson_basal.name, f"basal_0")
        curr_basal.select_set(True)
        bpy.context.view_layer.objects.active = curr_basal
        # Connect basal region to valve (Only one basal region necessary for porous medium).
        connect_valve_orifice(context, "Mitral", valve_index = 0) 
        basal_regions.append(curr_basal)
    # Interpolated mitral valve.
    elif context.scene.approach == 5:
        # Create reference basal region to derive node-connectivity in the valve orifice region from.
        bpy.ops.object.mode_set(mode='OBJECT')
        curr_basal = copy_object(poisson_basal.name, f"basal_ref")
        curr_basal.select_set(True)
        bpy.context.view_layer.objects.active = curr_basal
        deselect_object_vertices(curr_basal)
        bpy.ops.object.mode_set(mode='EDIT')
        # Connect basal region to valve.
        edges_reference = connect_valve_orifice_reference(context, "Mitral", valve_index = 1) 
        for i in range(5): # Use reference connection for the other basal regions.
            # Copy basal region and prepare it for mitral valve insertion.
            bpy.ops.object.mode_set(mode='OBJECT')
            curr_basal = copy_object(poisson_basal.name, f"basal_{i}")
            curr_basal.select_set(True) 
            bpy.context.view_layer.objects.active = curr_basal
            deselect_object_vertices(curr_basal)
            bpy.ops.object.mode_set(mode='EDIT')
            # Connect basal region to valve.
            connect_valve_orifice_from_reference(context, "Mitral", valve_index = i, edges_reference = edges_reference) 
            basal_regions.append(curr_basal)
    else: return False
    return basal_regions

def connect_valve_orifice_reference(context, valve_mode, valve_index): 
    """Connect orifices around valves with surrounding mesh nodes"""
    obj = bpy.context.active_object
    build_valve_surface(context, obj, valve_mode = valve_mode, ratio = 1, valve_index = valve_index) # Create valve interface nodes.
    bpy.ops.object.mode_set(mode='EDIT') # Change selection mode in edit mode for brige loop operator.
    selected_edges_before_indices, selected_edges_before_verts = get_selected_edges(obj)
    bpy.ops.mesh.looptools_bridge(cubic_strength=1, interpolation='linear', loft=False, loft_loop=False, min_width=100, mode='shortest', remove_faces=False, reverse=False, segments=1, twist=0)
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    selected_edges_after_indices, selected_edges_after_verts = get_selected_edges(obj)
    # Change to object mode 
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    # Save newly created edges
    new_edges = []
    new_edges_vert_indices = []
    for counter, e in enumerate(selected_edges_after_indices):
        if e not in selected_edges_before_indices:
            new_edges.append(e)
            new_edges_vert_indices.append(selected_edges_after_verts[counter])
        
    bpy.ops.object.mode_set(mode='OBJECT') 
    obj.select_set(False)
    return new_edges_vert_indices

def connect_valve_orifice_from_reference(context, valve_mode, valve_index, edges_reference): 
    """Connect the valve interface nodes with the valve orifice nodes from reference"""
    obj = bpy.context.active_object
    # Create exact valve nodes.
    build_valve_surface(context, obj, valve_mode = valve_mode, ratio = 1, valve_index = valve_index)
    bridge_edges_ventricle(obj, edges_reference)
    
def smooth_basal_region(context, basal, voxel_size):
    """Smooth basal region"""
    # Set current basal region active and go into edit mode.
    basal.select_set(True)
    bpy.context.view_layer.objects.active = basal
    select_only_inner_basal_vertices()
    bpy.ops.object.mode_set(mode='EDIT')
    # Merge close nodes and smooth them.
    if context.scene.approach == 3 or context.scene.approach == 4: bpy.ops.mesh.remove_doubles(threshold= voxel_size * 0.8, use_sharp_edge_from_normals=False, use_unselected=False) # Don't remove for approach 5
    bpy.ops.mesh.vertices_smooth(factor=0.75, repeat=10)
    # Select valve orifice edge loops for a better smoothing transition between valves and basal region.
    bpy.ops.object.vertex_group_set_active(group=str("Aortic_orifice"))
    bpy.ops.object.vertex_group_select()
    bpy.ops.object.vertex_group_set_active(group=str("Mitral_orifice"))
    bpy.ops.object.vertex_group_select()
    bpy.ops.object.vertex_group_set_active(group=str("lower_basal_edge_loop"))
    bpy.ops.object.vertex_group_deselect()
    # Smooth again with orifice selected.
    if context.scene.approach == 3 or context.scene.approach == 4: bpy.ops.mesh.remove_doubles(threshold=voxel_size * 0.9, use_sharp_edge_from_normals=False, use_unselected=False) # Don't remove for approach 5
    bpy.ops.mesh.vertices_smooth(factor=0.75, repeat=20)     
    # Relax lower orifice edge loop.
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.vertex_group_set_active(group=str("lower_basal_edge_loop"))
    bpy.ops.object.vertex_group_select()
    bpy.ops.mesh.looptools_relax(input='selected', interpolation='linear', iterations='5', regular=True)
    # Smooth inner vertices in an iterative process, removing small cells in the process.
    select_only_inner_basal_vertices()
    bpy.ops.object.mode_set(mode='EDIT')
    smooth_basal_iterations = 5
    for i in range(smooth_basal_iterations):
        if context.scene.approach == 3 or context.scene.approach == 4: bpy.ops.mesh.remove_doubles(threshold= voxel_size, use_sharp_edge_from_normals=False, use_unselected=False) # Don't remove for approach 5
        bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=10)
    bpy.ops.object.mode_set(mode='OBJECT') 
    basal.select_set(False)

def select_only_inner_basal_vertices():
    """Select all vertices in basal region except valve regions including the orifice edge loop and lower orifice loop"""
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.object.vertex_group_set_active(group=str("lower_basal_edge_loop"))
    bpy.ops.object.vertex_group_deselect()
    bpy.ops.object.vertex_group_set_active(group=str("AV"))
    bpy.ops.object.vertex_group_deselect()
    bpy.ops.object.vertex_group_set_active(group=str("MV"))
    bpy.ops.object.vertex_group_deselect()
    bpy.ops.object.mode_set(mode='OBJECT') 

class MESH_OT_connect_apical_and_basal(bpy.types.Operator):
    """Connect apical and basal region of ventricle"""
    bl_idname = 'heart.connect_apical_and_basal'
    bl_label = 'Connect apical and basal region of ventricle.'
    def execute(self, context):
        if not mesh_connect_apical_and_basal(context): return {'CANCELLED'}
        return {'FINISHED'} 
        
def mesh_connect_apical_and_basal(context):
    """Connect apical and basal region of ventricle"""
    cons_print("Connecting apical and basal regions...")
    selected_objects = context.selected_objects
    # Initialize names for basal regions.
    if context.scene.approach == 5: names = ["basal_0", "basal_1", "basal_2", "basal_3", "basal_4"]
    else: names = ["basal_0"]
    basal_regions = []
    # Set up basal regions so that the lower edge loop is selected.
    for name in names:
        if not name in bpy.data.objects: # Check if all necessary basal regions are present.
            cons_print(f"Missing following basal region: {name}")
            return False       
        else:  
            curr_basal = bpy.data.objects[name]
            basal_regions.append(curr_basal) # Add object to list of basal regions.
            # Unhide current basal region and use it as active object.
            curr_basal.hide_set(False)
            curr_basal.select_set(True)
            bpy.context.view_layer.objects.active = curr_basal
            # Select only lower basal edge loop vertex group.
            deselect_object_vertices(curr_basal)
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.object.vertex_group_set_active(group=str("lower_basal_edge_loop"))
            bpy.ops.object.vertex_group_select()
            bpy.ops.object.mode_set(mode='OBJECT')
            # Hide current basal region to improve solution speed as Blender does not need to render all objects at the same time.
            curr_basal.select_set(False)
            curr_basal.hide_set(True)
    # Create initial connection using a reference copy.
    reference = copy_object(bpy.types.Scene.reference_object_name, "reference")
    if not combine_apical_and_basal_region(context, basal_regions, reference, selected_objects): return False
    cleanup_basal_region(context) # Cleanup: Delete basal regions.
    return True

def combine_apical_and_basal_region(context, basal_regions, reference, selected_objects):
    """Combine the two regions by copying and joining the basal region for each ventricle and connecting the orifice edge loops between these newly joined objects"""
    # Deselect (and hide) all objects.
    for obj in selected_objects: 
        obj.select_set(False)  
        obj.hide_set(True)
    reference.select_set(False)
    ## Apply connecting operation for reference and save connecting edges used in the connection.
    prepare_geometry_for_bridging(reference, basal_regions[0]) # Prepare geometry for bridging by removing the original basal region and replacing it with the reconstructed basal region.
    edge_indices_bridge = bridge_edges_reference(context, reference) # Create initial connection between the upper apical and lower basal edge loop.
    inset_faces_smooth(context) # Refine connection by separating long connection faces into more uniformly sized faces.
    edge_indices_triangulate = triangulate_connection(True, reference, ref_edge_indices=[]) # Triangulate connection faces saving newly created edges.
    bpy.data.objects.remove(reference, do_unlink=True) # Cleanup: Remove reference object.
    # Compute the frame of the end diastole. Necessary for interpolated mitral valve.
    frame_EDV = round(context.scene.time_diastole / context.scene.time_rr *  context.scene.frames_ventricle) 
    # Compute volume list for computation of the intensity of smoothing of the connection between basal and apical region.
    volumelist = compute_volumes(selected_objects, False)
    if volumelist.index(min(volumelist)) > volumelist.index(max(volumelist)) or volumelist.index(min(volumelist)) != 0: cons_print(f"Warning: Ventricles not sorted.") # If list is not sorted
    ## Apply connecting-operation for remaining ventricle geometries.
    for counter, obj in enumerate(selected_objects):
        basal = basal_regions[get_valve_state_index(context, counter, frame_EDV)] # Choose basal region.
        # Apply connecting operation from reference.
        prepare_geometry_for_bridging(obj, basal) 
        bridge_edges_ventricle(obj, edge_indices_bridge)
        inset_faces_smooth(context)
        # Remove faces before triangulation.
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_more()
        bpy.ops.mesh.delete(type='ONLY_FACE') 
        bpy.ops.object.mode_set(mode='OBJECT')
        # Triangulate mesh.
        triangulate_connection(False, obj, edge_indices_triangulate)        
        # Add faces onto triangulation.
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.edge_face_add()
        bpy.ops.object.mode_set(mode='OBJECT')
        # Smooth connection dependent on which geometry is smoothed.
        smoothing_iter_factor = compute_smoothing_iteration_factor_connection(context, counter, volumelist)
        smooth_connection_and_basal_region(context, obj, smoothing_iter_factor)
        obj.hide_set(True)  
    for obj in selected_objects: obj.hide_set(False) # Cleanup: Unhide objects.
    return True

def get_valve_state_index(context, counter, frame_EDV):
    """Return the index of the basal region to be used for the given timestep"""
    if context.scene.approach == 3 or context.scene.approach == 4: return 0
    # Define the opening state of the mitral valve during multiple timeframes.
    begin_mvo = 4
    frames_mv_1 = [begin_mvo, frame_EDV-1] #  Fill with indices.
    frames_mv_2 = [begin_mvo + 1, frame_EDV-2]
    frames_mv_3 = [begin_mvo + 2, frame_EDV-3]
    frames_mv_4 = range(begin_mvo + 3, (frame_EDV-3))
    # Return index.
    if counter in frames_mv_1: return 1
    if counter in frames_mv_2: return 2
    if counter in frames_mv_3: return 3
    if counter in frames_mv_4: return 4
    else: return 0

def prepare_geometry_for_bridging(obj, final_basal_region):
    """Prepare the individual ventricle geometries for the bridging process. The original basal region is removed and the remaining apical region is joined into one object with the reconstructed basal region"""
    current_basal = copy_object(final_basal_region.name, 'temp') # Copy basal region for current object.
    # Change selection such that the copy of the basal region is merged into the input-object (obj).
    current_basal.select_set(False)
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    deselect_object_vertices(obj)
    # Cut basal part of the ventricle.
    for group in obj.vertex_groups:
        if group.name == "upper_apical_edge_loop":
            vg_orifice = group
            break
    subdivide_last_edge_loop(obj, vg_orifice) # Apply subdivide to smooth out further connection. 
    # Combine both geometries.
    current_basal.select_set(True)
    bpy.ops.object.join()

def bridge_edges_reference(context, reference):
    """Connect basal with apical part of the reference ventricle"""
    bpy.ops.object.mode_set(mode='EDIT') # Need to switch to edit mode. 
    selected_edges_before_indices, selected_edges_before_verts = get_selected_edges(reference) # Collect edge indices before operation.
    # Connect apical and basal region.
    bpy.context.tool_settings.mesh_select_mode = (False, True, False) # Activate edge mode in edit mode.
    bpy.ops.mesh.looptools_bridge(cubic_strength=1, interpolation='linear', loft=False, loft_loop=False, min_width=100, mode='shortest', remove_faces=False, reverse=False, segments=1, twist=context.scene.connection_twist) # Command used to connect two edge loops.
    bpy.context.tool_settings.mesh_select_mode = (True, False, False) # Return to vertex mode in edit mode.
    selected_edges_after_indices, selected_edges_after_verts = get_selected_edges(reference) # Save edge indices after operation.
    # Compare edges before and after operation and only keep the deviating edges and save their vertex indices in a list.
    new_edges, new_edges_vert_indices = ([] for i in range(2))
    for counter, e in enumerate(selected_edges_after_indices):
        if e not in selected_edges_before_indices:
            new_edges.append(e)
            new_edges_vert_indices.append(selected_edges_after_verts[counter])
    bpy.ops.object.mode_set(mode='OBJECT')
    return new_edges_vert_indices

def bridge_edges_ventricle(obj, new_edges_vert_indices): 
    """Connect basal with apical part of ventricle"""
    deselect_object_vertices(obj) # Deselect object vertices. This is necessary to add faces later in the function call.
    # Transfer data to edit-mode.
    bpy.ops.object.mode_set(mode='EDIT') 
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    for a, b in new_edges_vert_indices: # Create connecting edges between all vertex pairs a and b and keep them selected.
        bm.edges.new((bm.verts[a], bm.verts[b]))
        bm.verts[a].select = True
        bm.verts[b].select = True
    # Create faces between all selected connecting edges (Wireframe to Surface-mesh).
    bpy.ops.object.mode_set(mode='OBJECT') # Necessary switch between object and edit mode to update blender-object.
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.edge_face_add() # Create faces between connection edges.
    bpy.ops.object.mode_set(mode='OBJECT') 

def inset_faces_smooth(context):
    """Create new vertices along the connection between apical and basal region. This subdivision aims to more equally space the height and width of these faces"""
    distance = context.scene.min_valves - context.scene.remove_basal_threshold # Distance between lowest valve node and highest apical point of all ventricles after basal removal in z-direction.
    bpy.ops.object.mode_set(mode='EDIT')
    for counter, value in enumerate(range(context.scene.inset_faces_refinement_steps)):
        inset_thickness = distance / (4 * 2**(counter)) # Reduce the offset between newly added edge_loops by the factor 4.
        bpy.ops.mesh.inset(thickness= inset_thickness, use_even_offset=False, depth=0, use_select_inset=True) # Insert faces along the bridge between apical and basal.
        bpy.ops.mesh.vertices_smooth(factor=0.5, repeat= 1 + counter)  # Smooth connection between apical and basal region including the newly added faces.
    bpy.ops.object.mode_set(mode='OBJECT')
  
def triangulate_connection(bool_ref, obj, ref_edge_indices):
    """Triangulate connection between apical and basal region"""
    edges_vert_indices_tri = []
    if bool_ref: # Initial triangulation as a reference.
        bpy.ops.object.mode_set(mode='EDIT') 
        selected_edges_before_indices, selected_edges_before_verts = get_selected_edges(obj) # Collect edge indices before operation.
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY') # Triangulate connection.
        selected_edges_after_indices, selected_edges_after_verts = get_selected_edges(obj) # Collect edge indices after operation.
        # Compare edges before and after operation and only keep the new edges and save their corresponding vertex indices in a list.
        for counter, e in enumerate(selected_edges_after_indices):
            if e not in selected_edges_before_indices:
                edges_vert_indices_tri.append(selected_edges_after_verts[counter])
        bpy.ops.object.mode_set(mode='OBJECT') 
    else: # Apply triangulation from reference.
        deselect_object_vertices(obj) # Deselect all faces to add faces later.
        bpy.ops.object.mode_set(mode='EDIT') 
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        # Create connecting edge between a and b
        for a, b in ref_edge_indices:
            bm.edges.new((bm.verts[a], bm.verts[b]))
            bm.verts[a].select = True
            bm.verts[b].select = True
        # Create faces between connecting edges
        bpy.ops.object.mode_set(mode='OBJECT') # Necessary to update geometry in Blender.
        bpy.ops.object.mode_set(mode='EDIT') 
        bpy.ops.object.mode_set(mode='OBJECT') # Change to object mode.
    return edges_vert_indices_tri

def compute_smoothing_iteration_factor_connection(context, counter, volumelist):
    """Compute how strongly the connection between basal and apical region will be smoothed."""
    max_index = volumelist.index(max(volumelist)) # Index of EDV in volume list.
    min_index = volumelist.index(min(volumelist)) # Index of ESV in volume list.
    if counter >= min_index and counter <= max_index:
        smoothing_iter_factor = 1  - (counter - min_index) * 1 / (max_index - min_index)
    elif counter > max_index:
        smoothing_iter_factor = 1  * (counter - max_index) / (len(volumelist)- max_index)
    else: 
        cons_print(f"Mistake in computation of smoothing iteration factor for the smoothing of the basal-apical connection.")
        smoothing_iter_factor = 1
    smoothing_iter_factor = smoothing_iter_factor * 2 # Increase smoothing iteration factor by factor 2.
    if context.scene.approach == 5: smoothing_iter_factor = smoothing_iter_factor * 2 # Further increase it for approach 5 as it has a higher vertex density in the basal region.
    return smoothing_iter_factor

def smooth_connection_and_basal_region(context, obj, smoothing_iter_factor): 
    """Smooth basal ventricle region excluding the valves"""
    deselect_object_vertices(obj) # Reset node selection.
    # Select all vertices between the lowest valve vertex and the highest basal region vertex.
    bm = transfer_data_to_mesh(obj)
    for v in bm.verts: # Select all vertices above highest apical vertex in z-direction.
        vertex_coords = obj.matrix_world @ v.co 
        if vertex_coords[2] >= context.scene.remove_basal_threshold: v.select = True 
    bm.to_mesh(obj.data)
    bpy.ops.object.mode_set(mode='EDIT') 
    # Select all nodes inside the connection.
    bpy.ops.object.vertex_group_set_active(group=str("lower_basal_edge_loop"))
    bpy.ops.object.vertex_group_select()  
    bpy.ops.object.vertex_group_set_active(group=str("upper_apical_edge_loop"))
    bpy.ops.object.vertex_group_select()  
    # Select edge loops below the connection. Selecting all apical nodes would greatly shrink the ventricle volume in that region.   
    for i in range(context.scene.sm_reps):  # Iteratively smooth the selected nodes. This especially smooths the transition between connection and apical nodes.
        bpy.ops.mesh.select_more() # Select edge loops until reaching an edgeloop, that was not subdivided during the removal of the basal region.
        # Exclude valve nodes in the selection process.
        bpy.ops.object.vertex_group_set_active(group=str("AV"))
        bpy.ops.object.vertex_group_deselect()
        bpy.ops.object.vertex_group_set_active(group=str("MV"))
        bpy.ops.object.vertex_group_deselect()
        if i == 0:
            smooth_iter = round(context.scene.max_con_sm_iter * smoothing_iter_factor) + context.scene.min_con_sm_iter # Strong smoothing initially.
        else:
            smooth_iter = round(context.scene.max_con_sm_iter / 5 * smoothing_iter_factor * (context.scene.sm_reps-i)) + context.scene.min_con_sm_iter # Weaker smoothing after first iteration. As hard smoothing creates kinks between smoothed nodes and unsmoothed nodes.
        bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=smooth_iter)
    bpy.ops.object.mode_set(mode='OBJECT')

class MESH_OT_Ventricle_Sort(bpy.types.Operator): 
    """Sort ventricles by volume starting with ESV"""
    bl_idname = 'heart.sort_ventricles'
    bl_label = 'Sort ventricles by volume starting with ESV.'
    def execute(self, context):
        if not sort_ventricles(context.selected_objects): return{'CANCELLED'}
        return{'FINISHED'}

def sort_ventricles(selected_objects):
    """Sort ventricles by volume starting with minimal volume (End-systolic volume)"""
    cons_print(f"Sorting ventricle geometries starting in End-systolic state.")
    volumes = compute_volumes(selected_objects, False)
    val, idx = min((val, idx) for (idx, val) in enumerate(volumes))
    # Free up names by renaming the objects to temporary names.
    for counter, obj in enumerate(selected_objects): obj.name = f"temp_ven_{counter}" 
    # Re-arrange names of ventricles.
    for old_index, obj in enumerate(selected_objects): 
        if old_index < idx: new_index = old_index - idx + len(volumes) # If index would go below 0 revert to the other side of the list.
        else:new_index = old_index - idx 
        obj.name = f"ventricle_{new_index}" # Rename selected objects.

class MESH_OT_Ventricle_Interpolation(bpy.types.Operator): 
    """(Linearly) Interpolate ventricle geometry to a larger amount of timesteps"""
    bl_idname = 'heart.ventricle_interpolation'
    bl_label = 'Interpolate selected objects.'
    def execute(self, context):
        if not interpolate_ventricle(context): return{'CANCELLED'}
        return{'FINISHED'}
    
def interpolate_ventricle(context):
    """Interpolate selected objects to n(frames_ventricle) objects"""
    cons_print("Interpolating ventricle geometries...")
    selected_objects = context.selected_objects
    for counter, obj in enumerate(selected_objects): obj.name = f"z_ven_{counter}" # Rename selected objects.
    # Compute timesteps before and after interpolation.
    timestep_original = context.scene.time_rr / len(selected_objects) 
    timestep_int = context.scene.time_rr / (context.scene.frames_ventricle) 
    int_objects = []
    # Copy first object and add it to the end of the object list to be able to interpolate between the last timestep and the first.
    initial_obj = copy_object(selected_objects[0].name, f"z_ven_{len(selected_objects)}")
    selected_objects.append(initial_obj)
    for i in range(context.scene.frames_ventricle): 
        if i == 0: # Exceptions for first object, since it doesn not have to be interpolated.
           int_objects.append(copy_object(selected_objects[0].name, 'ventricle_0')) 
           continue # Skip further functionality and fo straight to the second iteration.
        # Compute indices of the objects used in the current interpolation.
        current_time = (i) * timestep_int
        object_one_index = math.floor(current_time / timestep_original) # Find first object in the current timestep.
        object_one = selected_objects[object_one_index]
        object_two = selected_objects[object_one_index + 1]
        # Interpolate chosen objects using an interpolation factor defining how strongly each object influences the resulting object after interpolation.
        int_factor = (current_time - (object_one_index * timestep_original)) / timestep_original 
        int_obj = interpolate_two_object(object_one, object_two, int_factor, new_obj_index=i) 
        # Add the interpolated object to the list of interpolated objects.
        if not int_obj: # Exception when object could not be interpolated.
            cons_print(f"Problem during interpolation of timestep: {i}.")
            return False
        int_obj.select_set(True)
        int_objects.append(int_obj)
    for i in range(len(selected_objects)): bpy.data.objects.remove(bpy.data.objects[f"z_ven_{i}"], do_unlink=True) # Cleanup: Delete original objects.
    return int_objects

def interpolate_two_object(object_one, object_two, int_factor, new_obj_index):
    """Interpolate two objects using an interpolation factor"""
    # Get list of node coordinates for both objects.
    coords_one = get_object_vertices(object_one)
    coords_two = get_object_vertices(object_two)
    if len(coords_one) != len(coords_two): # Check if both objects have the same amount of nodes.
        cons_print("Tried interpolating objects with deviating amount of nodes.")
        return False
    # Compute interpolation for all nodes.
    int_coords = []
    for index in range(len(coords_one)):
        vec_int = coords_one[index] + (coords_two[index] - coords_one[index]) * int_factor # Interpolation of a single node.
        int_coords.append(vec_int)
    # Create new object by copying the first object and change the nodes coordinates to the interpolated node coordinates.
    int_obj = copy_object(object_one.name, f"ventricle_{new_obj_index}") # Copy object_one to create object between original timeframes.
    bpy.context.view_layer.objects.active = int_obj # Set new object as active object.
    change_object_vertices(int_obj, int_coords) # Change node coordinates.
    return int_obj
    
def get_object_vertices(obj):  
    """Return list of vertex coordinates of an object"""
    if obj.mode == 'EDIT': bpy.ops.object.mode_set() # Change to object mode if in edit mode to be able to use obj.data.vertices.
    coords = []
    for v in obj.data.vertices: coords.append(obj.matrix_world @ v.co)
    return coords

def change_object_vertices(obj, vertices):
    """Change object vertices with a list of vertices"""
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data) # Transfer data to mesh.
    for v in bm.verts: v.co = vertices[v.index] # Change vertices.
    bpy.ops.object.mode_set(mode='OBJECT')

class MESH_OT_Add_Vessels_Valves(bpy.types.Operator):
    """Create objects for aorta, atrium and valves"""
    bl_idname = 'heart.add_vessels_valves'
    bl_label = 'Create objects for aorta, atrium and valves.'
    def execute(self, context):
        add_vessels_and_valves(context)
        return{'FINISHED'}
    
def add_vessels_and_valves(context): 
    """Create objects for aorta, atrium and valves"""
    add_aorta(context)
    add_atrium(context)
    # Porous mitral valve are only inserted in the porous medium approach. For the interpolated mitral valve approach the mitral valve becomes part of the ventricle. Aortic valve is not inserted in A3.
    if context.scene.approach == 4 or context.scene.approach == 5: create_porous_valve_zones(context, 'Aortic', ['A4_AV_imperm', 'A4_AV_perm', 'A4_AV_res']) 
    if context.scene.approach == 4: create_porous_valve_zones(context, 'Mitral', ['A4_MV_imperm', 'A4_MV_perm', 'A4_MV_res']) 

def add_atrium(context):
    """Copy atrium and place it above the mitral valve as a separate object"""
    # Choose the atrium fitting the current case.
    atrium = copy_object(f"A{context.scene.approach}_Atrium", "atrium")
    atrium.select_set(True)
    bpy.context.view_layer.objects.active = atrium
    scale_rotate_translate_object(context, atrium, "Mitral", ratio=1)
    return atrium
   
def add_aorta(context):
    """Copy aorta and place it above the aortic valve as a separate object"""
    aorta = copy_object(f"A{context.scene.approach}_Aorta", "aorta")
    aorta.select_set(True)
    bpy.context.view_layer.objects.active = aorta
    scale_rotate_translate_object(context, aorta, "Aortic", ratio=1)
    return aorta

def create_porous_valve_zones(context, valve_mode, valve_strings):
    """Add separate porous zone objects (valves, atrium and aorta) into workspace"""
    # Set up object name after insertion.
    if valve_mode == 'Aortic':
        valve_substring = "av"
    elif valve_mode == 'Mitral':
        valve_substring = "mv"
    else: return False
    obj_types = ["_perm", "_res", "_imperm"] # Collection of valve types for name setup.
    for obj_str in valve_strings:
        for obj_type in obj_types:
            if obj_type in obj_str: new_name = f"{valve_substring}{obj_type}" # Combine substrings to string of new object.
        # Create new object by copying it from existing object.
        new_obj = copy_object(obj_str, new_name)
        new_obj.select_set(True)
        bpy.context.view_layer.objects.active = new_obj
        scale_rotate_translate_object(context, new_obj, valve_mode = valve_mode, ratio = 1)

class MESH_OT_Quick_Recon(bpy.types.Operator):
    """Quick geometrical reconstruction of all ventricles containing all steps of the reconstruction algorithm in one execution"""
    bl_idname = 'heart.quick_recon'
    bl_label = 'Quick geometrical reconstruction of all ventricles containing all steps of the reconstruction algorithm in one execution.'
    def execute(self, context):
        if context.scene.approach == 5:
            if not interpolate_ventricle(context): return{'CANCELLED'} # Interpolate ventricle geometry.
        remove_multiple_basal_region(context) # Remove old basal region.
        if not mesh_create_basal(context): return{'CANCELLED'}# Operations to create basal region of the ventricle containing valve orifices.
        if not mesh_connect_apical_and_basal(context): return {'CANCELLED'} # Connect apical regions with corresponding bassal regions.
        add_vessels_and_valves(context) # Add surrounding objects including aorta, atrium and valves.
        return{'FINISHED'} 

def cleanup_basal_region(context):
    """Remove basal regions as part of the reconstruction pipeline cleanup"""
    if context.scene.approach == 3 or context.scene.approach == 4: amount_basal_regions=1
    else: amount_basal_regions=5
    for i in range(amount_basal_regions): bpy.data.objects.remove(bpy.data.objects["basal_" + str(i)], do_unlink=True)
    
class MESH_DEV_volumes(bpy.types.Operator):  
    """Compute volumes of selected objects"""
    bl_idname = 'heart.dev_volumes'
    bl_label = 'Compute volumes of selected objects.'
    def execute(self, context):
        compute_volumes(bpy.context.selected_objects, True)
        return{'FINISHED'}

def compute_volumes(objects, bool_print):
    """Print each volume and surface area of a given list of objects"""
    volumelist = []
    for obj in objects:
        volume, area = compute_volume_area(obj)
        volumelist.append(volume)
        if bool_print: cons_print(f"{obj.name} with volume: {round(volume/1000, 4)} ml and surface area: {round(area/100, 4)} mm^2.")
    if bool_print:
        cons_print(f"Max volume (EDV): {max(volumelist)/1000} ml at position: {volumelist.index(max(volumelist))}")  
        cons_print(f"Min volume (ESV): {min(volumelist)/1000} ml at position: {volumelist.index(min(volumelist))}")  
        cons_print(f"Stroke volume: {(max(volumelist)-min(volumelist))/1000} ml")
        cons_print(f"Volumelist: {volumelist}")
    return volumelist

def compute_volume_area(obj):
    """Return volume and surface area of object of object"""  
    # Transfer object into mesh.
    bm = transfer_data_to_mesh(obj)
    # Compute volume and surface area.
    volume = bm.calc_volume(signed=True) 
    area = sum(f.calc_area() for f in bm.faces)
    return volume, area

class MESH_DEV_indices(bpy.types.Operator):
    """Print indices of selected nodes and print the total number of selected nodes"""
    bl_idname = 'heart.dev_indices'
    bl_label = 'Print indices of selected nodes and print the total number of selected nodes.'
    def execute(self, context):
        if not get_selected_objects_vertices(context): return{'CANCELLED'}
        return{'FINISHED'}
    
def get_selected_objects_vertices(context):
    """Return indices of currently selected vertices"""
    if bpy.context.mode != 'EDIT': bpy.ops.object.mode_set(mode='EDIT') # Toggle to edit mode.
    obj = context.object
    bm = bmesh.from_edit_mesh(obj.data)
    verts = []
    for v in bm.verts:
        if v.select:
            vertice_coords = obj.matrix_world @ v.co # Transform to global coordinate system.
            cons_print(f"Vertex {v.index} at ({round(vertice_coords.x, 2)}|{round(vertice_coords.y, 2)}|{round(vertice_coords.z, 2)})")
            verts.append(v)
    cons_print(f"Amount of selected vertices: {len(verts)}")
    return verts

class MESH_DEV_edge_index(bpy.types.Operator): 
    """Print edge index between two selected indices"""
    bl_idname = 'heart.dev_check_edges'
    bl_label = 'Print edge index between two selected indices.'
    def execute(self, context):
        if not check_edge(context): return{'CANCELLED'}
        return{'FINISHED'}

def check_edge(context):
    """Print edge index between two selected indices"""
    # Read selected vertices.
    obj = context.edit_object
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    if hasattr(bm.verts, "ensure_lookup_table"): bm.verts.ensure_lookup_table()
    verts = [v for v in bm.verts if v.select] # Get selected vertices.
    # Only two vertices may be selected.
    if len(verts) != 2:
        cons_print(f"Select exactly two vertices.")
        return False
    # Check edge connectivity between vertices and print out result.
    V1 = verts[0] # Define vertices one and two.
    V2 = verts[1]
    for e in V1.link_edges: 
        if e.other_vert(V1) is V2:
            cons_print(f"Vertex indices : {V1.index}, {V2.index} with edge index: {e.index}.")
            return e.index
    cons_print(f"Nodes not connected.")
    return False

class MESH_OT_check_node_connectivity(bpy.types.Operator):
    """Check node connectivity between all selected objects"""
    bl_idname = 'heart.dev_check_node_connectivity'
    bl_label = 'Check node connectivity between all selected objects.'
    def execute(self, context):
        check_node_connectivity(context)
        return{'FINISHED'}

def check_node_connectivity(context):
    """Check node-connectivity between all selected objects"""
    vertices, edges, faces = ([] for i in range(3))
    for i, obj in enumerate(context.selected_objects):
        deselect_object_vertices(obj)
        # Check for vertices with only two neighbour vertices.
        me = obj.data
        bpy.ops.object.mode_set(mode='EDIT') 
        bm = bmesh.from_edit_mesh(me)
        # Select vertices in edge loop with only two connecting vertices.
        for v in bm.verts:
            if len(get_neighbour_vertices(v)) <= 2: 
                cons_print(f"Found a vertex with only 2 neighbour vertices in object {obj.name} with vertex index {v.index} at position {obj.matrix_world @ v.co}")
                v.select = True
                return False
        # Return to object mode and update the mesh to the obeject.
        bm.select_flush_mode()   
        me.update()
        bpy.ops.object.mode_set(mode='OBJECT') 
    # Initialize the list of vertices, edges and faces each with their respective connecting vertices.
        if i == 0:
            for v in obj.data.vertices: vertices.append(v.index)
            for e in obj.data.edges: edges.append([e.index, e.vertices[0] , e.vertices[1]])
            for f in obj.data.polygons: faces.append([f.index, f.vertices[0], f.vertices[1], f.vertices[2]])
        else:
    # Checks for the length of vertices, edges and faces.
            # Check if the amount of vertices matches.
            if len(obj.data.vertices) != len(vertices):
                cons_print(f"Object: {obj.name} has mismatching amount of vertices.")
                return False
            # Check if the amount of edges matches.
            if len(obj.data.edges) != len(edges):
                cons_print(f"Object: {obj.name} has mismatching amount of edges.")
                return False
            # Check if the amount of faces matches.
            if len(obj.data.polygons) != len(faces):
                cons_print(f"Object: {obj.name} has mismatching amount of faces.")
                return False
    # Connectivity checks. 
            # Face-connectivity check.
            for counter, f in enumerate(obj.data.polygons):
                if not (faces[counter][0] == f.index and faces[counter][1]== f.vertices[0] and faces[counter][2] ==  f.vertices[1] and faces[counter][3] ==  f.vertices[2]):
                    cons_print(f"Face mismatch in object: {obj.name} at face {f.index} with face-center at {f.center}.") 
                    f.select = True # Select problematic face.
                    return False
            # Edge-connectivity check.
            for counter, e in enumerate(obj.data.edges):
                if not (edges[counter][0] == e.index and edges[counter][1]== e.vertices[0] and edges[counter][2] ==  e.vertices[1]):
                    cons_print(f"Edge mismatch in object: {obj.name} at edge {e.index}.")
                    e.select = True # Select problematic edge.
                    return False  
    cons_print(f"Node-connectivity matched for all {len(context.selected_objects)} selected elements.")
    return True

class MESH_OT_ApproachSelection(bpy.types.Operator):
    """Select approach"""
    bl_label = "Select approach"
    bl_idname = "wm.approach_selection"
    
    approach_enum : bpy.props.EnumProperty(
        name= "Select approach",
        description= "Select approach",
        items= [('Op_A3', 'Porous jump (A3)', 'Interpolated MV (A5)'),
                ('Op_A4', 'Porous Medium (A4)', 'Interpolated MV (A5)'),
                ('Op_A5', 'Interpolated MV (A5)', 'Interpolated MV (A5)')
                ]
    ) 
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "approach_enum")
        
    def execute(self, context):
        if self.approach_enum == "Op_A3": context.scene.approach = 3
        if self.approach_enum == "Op_A4": context.scene.approach = 4
        if self.approach_enum == "Op_A5": context.scene.approach = 5
        return {'FINISHED'} 

class MESH_DEV_color_min_dist(bpy.types.Operator):
    """Color all faces of the currently selected object with the minimal distance to a reference object 'ref_obj'. The colors range from blue to red (0-1) and are normalized with the maximum over all minimal distances"""
    bl_idname = 'heart.color_min_dist'
    bl_label = 'Color an object with its facewise minimal distance to a reference object'
    def execute(self, context):
        if not color_min_dist(context): return{'CANCELLED'}
        return{'FINISHED'}
    
def compute_colors(distances):
    """Compute face colors for the normalized distance value."""
    colors = []
    for dist in distances:
        if dist <= 0.5: colors.append([ 2 * dist , 2 * dist  , 1   , 1 ])
        else: colors.append([ 1 , 1 - (2 * dist - 1) , 1 - (2 * dist - 1) , 1 ])
    return colors
    
def get_face_centers_of_object(obj):
    """Return all face centers of an object as a numpy vector."""
    obj_face_centers = []
    long_shift = obj.get("long_shift", 0)
    if long_shift == 0: # Object without longitudinal shift
        for obj_face in obj.data.polygons:
            objFacePos = obj_face.center
            face_vector_np = np.array([objFacePos.x, objFacePos.y, objFacePos.z])
            obj_face_centers.append(face_vector_np)
    else: # Object with longitudinal shift
        for obj_face in obj.data.polygons:
            objFacePos = obj_face.center
            face_vector_np = np.array([objFacePos.x, objFacePos.y, objFacePos.z - long_shift])
            obj_face_centers.append(face_vector_np)
    return obj_face_centers

def compute_norm_face_distances(ico_obj_face_centers, other_obj_face_centers): 
    """Compute the normalized minimal distance for each face of on object to any dace of another object"""
    distances = []
    # Compute all distances minimum face distances between one face of a chosen abject to any face of another object
    for ico_face_center in ico_obj_face_centers: distances.append(compute_min_face_distance(ico_face_center, other_obj_face_centers))
    max_dist = max(distances)
    cons_print(f"Maximum distance: {max_dist}")
    for index, dist in enumerate(distances): distances[index] = dist / max_dist # Normalize distances.
    return distances

def compute_min_face_distance(ico_face_center, other_obj_face_centers):
    """Find minimum distance of a given face to any face of a given body."""
    dist = 1000000
    for other_face_center in other_obj_face_centers:
        curr_dist = compute_distance(ico_face_center, other_face_center)
        if curr_dist < dist: dist = curr_dist
    return dist

# Njit decorator to run this at machine speed
@nb.njit
def compute_distance(ico_face_center, other_face_center):
    """Efficiently compute distance between two vectors"""
    return np.linalg.norm(ico_face_center - other_face_center)

def color_min_dist(context):
    """Color an object with its facewise minimal distance to a reference object"""
    cons_print(f"Coloring minimal face distance for obj.")
    # Deselect object vertices and faces
    if context.active_object: ico_object = context.active_object
    else: 
        cons_print(f"No active object. Terminating coloring function.")
        return False
    deselect_object_vertices(context.active_object) # Deselect vertices
    ico_object.data.materials.clear()# Clear material slots
    # Switch to edit mode
    bpy.ops.object.editmode_toggle()
    # Set geometry data from mesh object
    ico_bmesh = bmesh.from_edit_mesh(ico_object.data)
    other_obj = bpy.data.objects["ref_obj"]
    # Create list of the face centers of both objects.
    ico_obj_face_centers = get_face_centers_of_object(ico_object)
    other_obj_face_centers = get_face_centers_of_object(other_obj)
    # Compute colors and normalize them
    distances = compute_norm_face_distances(ico_obj_face_centers, other_obj_face_centers)
    colors = compute_colors(distances)
    # Iterate through each face of the mesh to color the faces
    for index, face in enumerate(ico_bmesh.faces):
        # create a new material
        mat = bpy.data.materials.new(name=f"face_{face.index}")
        mat.diffuse_color = colors[index]
        # Add the material to the object and set it active
        ico_object.data.materials.append(mat)
        ico_object.active_material_index = face.index
        # Select the face and assign the active material
        face.select = True
        bpy.ops.object.material_slot_assign()
        face.select = False
    # Turn off edit mode
    bpy.ops.object.editmode_toggle()

class MESH_DEV_test(bpy.types.Operator):
    """Test function for development"""
    bl_idname = 'heart.test'
    bl_label = 'Empty test function'
    def execute(self, context):
        test_function(context)
        return{'FINISHED'}

def test_function(context):
    """Empty test function"""
    cons_print(f"Running test function")

#-----

#PANELS

#-----

class PANEL_Files(bpy.types.Panel):
    bl_label = "File Management"
    bl_idname = "PT_File_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GVR-Pipeline'
    bl_option = {'DEFAULT_CLOSED'}
    def draw(self, context):
        layout = self.layout

        # Import button
        row = layout.row()
        layout.operator('heart.import_ventricle', text= "Import ventricles", icon= "IMPORT")

        # Import folder path selector
        row = layout.row()
        row.prop(context.scene, "ventricle_import_dir", text="Import folder")
        
        # Quick reset
        row = layout.row()
        layout.operator('heart.quick_reset', text="Quick reset", icon="FILE_REFRESH")

        # Export button
        row = layout.row()
        layout.operator('heart.export_ventricle', text= "Export ventricle", icon = 'EXPORT')

        # Export folder path selector
        row = layout.row()
        row.prop(context.scene, "ventricle_export_dir", text="Export folder")

class PANEL_Position_Ventricle(bpy.types.Panel):
    bl_label = "Ventricle position (mm)"
    bl_idname = "PT_Ventricle"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GVR-Pipeline'
    bl_option = {'DEFALUT_CLOSED'}
    def draw(self, context):
        layout = self.layout 
        row = layout.row()
        row.prop(context.scene, 'pos_top', text="Basal (Top)")  
        row = layout.row()
        props = layout.operator('heart.get_point', text= "Select basal node")
        props.point_mode = "Top"
        row = layout.row()
        row.prop(context.scene, 'pos_bot', text="Apex (bottom)")    
        row = layout.row()
        props = layout.operator('heart.get_point', text= "Select apex node")
        props.point_mode = "Bot"
        row = layout.row()
        row.prop(context.scene, 'pos_septum', text="Septum (side)")  
        row = layout.row()
        props = layout.operator('heart.get_point', text= "Select node at septum")
        props.point_mode = "Septum"  
        row = layout.row()
        row = layout.row()
        layout.operator('heart.ventricle_rotate', text= "Translate and rotate", icon = 'CON_ROTLIKE')

class PANEL_Valves(bpy.types.Panel):
    bl_label = "Valve options"
    bl_idname = "PT_Valves"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GVR-Pipeline'
    bl_option = {'DEFALUT_CLOSED'}
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.label(text= "Mitral valve", icon = 'META_ELLIPSOID')  
        row = layout.row()
        row.prop(context.scene, 'translation_mitral', text="Translation(mm)", icon='META_ELLIPSOID')
        row = layout.row()
        row.prop(context.scene, 'angle_mitral', text="Angle", icon='META_ELLIPSOID')
        row = layout.row()
        row.prop(context.scene, 'mitral_radius_long', text="Long mitral radius", icon='META_ELLIPSOID')
        row = layout.row()
        row.prop(context.scene, 'mitral_radius_small', text="Small mitral radius")
        row = layout.row()
        row.label(text= "Aortic valve", icon = 'MESH_CIRCLE')   
        row = layout.row()
        row.prop(context.scene, 'translation_aortic', text="Translation(mm)")
        row = layout.row()
        row.prop(context.scene, 'angle_aortic', text="Angle")
        row = layout.row()
        row.prop(context.scene, 'aortic_radius', text="Aortic radius", icon='META_BALL')
        if dev_env_tools:
            # Calculate Valve Diameter
            row = layout.row()
            row.operator('heart.calculate_valve_diameter', text = "Calculate Diameter", icon = "MESH_CIRCLE")
            row = layout.row()
            layout.operator('heart.build_valve',  text= "Add valve interface nodes", icon = 'PROP_OFF')
            row = layout.row()
            layout.operator('heart.support_struct',  text= "Build support structure around valves", icon = 'PROP_ON')
                
class PANEL_Poisson(bpy.types.Panel): 
    bl_label = "Poisson Surface Reconstruction"
    bl_idname = "PT_Poisson"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GVR-Pipeline'
    bl_option = {'DEFALUT_CLOSED'}
    def draw(self, context):
        layout = self.layout      
        row = layout.row()
        layout.operator('heart.poisson', text= "Apply Poisson surface reconstruction", icon = 'PROP_ON')
        row = layout.row()
        layout.operator('heart.create_valve_orifice', text= "Create valve orifices", icon = 'ALIASED')
        row = layout.row()
        layout.operator('heart.connect_valve', text= "Connect valves to orifices", icon = 'ANTIALIASED')
   
class PANEL_Setup_Variables(bpy.types.Panel):
    bl_label = "Algorithm setup variables"
    bl_idname = "PT_test"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GVR-Pipeline'
    bl_option = {'DEFALUT_CLOSED'}
    def draw(self, context):
        layout = self.layout   
        row = layout.row()
        row.prop(context.scene, 'remove_basal_threshold', text="Threshold for basal region removal") 
        row = layout.row()
        row.prop(context.scene, 'mean_reference', text="Use mean instead of max volume as reference") 
        row = layout.row()
        row.label(text= "Interpolation variables") 
        row = layout.row()
        row.prop(context.scene, 'time_rr', text="Time RR-duration") 
        row = layout.row()
        row.prop(context.scene, 'time_diastole', text="Time diastole") 
        row = layout.row()
        row.prop(context.scene, 'frames_ventricle', text="Frames after interpolation") 
        row = layout.row()
        row.label(text= "Advanced setup variables", icon = 'HOME') 
        row = layout.row()
        row.prop(context.scene, 'poisson_depth', text="Depth of poisson reconstruction algorithm") 
        row = layout.row()
        row.label(text= "Connection initialization variables") 
        row = layout.row()
        layout.prop(context.scene, "connection_twist", text="Twist during connecting algorithm")
        row = layout.row()
        layout.prop(context.scene, "inset_faces_refinement_steps", text="Refinement steps for insetting faces")
        row = layout.row()
        row.label(text= "Connection smoothing variales") 
        row = layout.row()
        layout.prop(context.scene, "max_con_sm_iter", text="Maximum smoothing iterations")
        row = layout.row()
        layout.prop(context.scene, "min_con_sm_iter", text="Minimum smoothing iterations")
        row = layout.row()
        layout.prop(context.scene, "sm_reps", text="Smoothing repetitions")
      
class PANEL_Pipeline(bpy.types.Panel):
    bl_label = "Geometric ventricle reconstruction pipeline"
    bl_idname = "PT_Pipeline"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GVR-Pipeline'
    bl_option = {'DEFALUT_CLOSED'}
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.label(text= "Setup pipeline", icon = 'SETTINGS')
        row = layout.row()
        layout.operator('heart.sort_ventricles', text= "Sort volumes", icon = 'EVENT_F1')
        row = layout.row()
        row.label(text= "Setup ventricle position and rotation", icon = 'EVENT_F2')
        row = layout.row()
        row.label(text= "Setup valves", icon = 'EVENT_F3')
        row = layout.row()
        row.label(text= "Setup algorithm variables", icon = 'EVENT_F4')
        row = layout.row()
        row.label(text= f"Current approach: A{context.scene.approach}")
        row = layout.row()
        layout.operator("wm.approach_selection", icon = 'EVENT_F5')
        row = layout.row()
        if context.scene.approach == 5:
            layout.operator('heart.ventricle_interpolation', text= "Interpolate ventricle", icon = 'EVENT_F6')
            row = layout.row()
        row.label(text= "Run pipeline", icon = 'PLAY')
        row = layout.row()
        row.operator('heart.remove_basal', text= "Remove basal region", icon = 'LIBRARY_DATA_OVERRIDE')  
        row = layout.row()
        layout.operator('heart.create_basal', text= "Create basal region", icon = 'SPHERECURVE')
        row = layout.row()
        layout.operator('heart.connect_apical_and_basal', text= "Connect basal and apical regions", icon = 'ORPHAN_DATA')
        row = layout.row()
        layout.operator('heart.add_vessels_valves', text= "Add atrium, aorta and valves", icon = 'HANDLE_AUTO')
        row = layout.row()
        layout.operator('heart.quick_recon', text= "Quick reconstruction", icon = 'HEART')

class PANEL_Dev_tools(bpy.types.Panel):
    bl_label = "Development tools"
    bl_idname = "PT_Dev_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GVR-Pipeline'
    bl_option = {'DEFAULT_CLOSED'}
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        layout.operator('heart.dev_volumes', text= "Compute volumes", icon = 'MESH_CUBE')
        row = layout.row()
        layout.operator('heart.dev_indices', text= "Get vertex indices", icon = 'THREE_DOTS')
        row = layout.row()
        layout.operator('heart.dev_check_edges', text= "Get edge index", icon = 'ARROW_LEFTRIGHT')
        row = layout.row()
        layout.operator('heart.dev_check_node_connectivity', text= "Node-connectivity check", icon = 'MOD_PARTICLES') 
        # Color minimal distance button
        row = layout.row()
        layout.operator('heart.color_min_dist', text= "Color minimal distance to raw object", icon = 'COLORSET_04_VEC') 
        
        row = layout.row()
        layout.label(text='Comparison of volume curves')
        # Plot STL setting file
        row = layout.row()
        row.prop(context.scene, "plot_input_path", text = "Plot inputs")
        # Plot STL and display
        row = layout.row(align=True)
        row.operator('heart.plot_stl', text = "Show", icon = 'AXIS_FRONT')
        row.operator('heart.save_plot_stl', text = 'Save', icon = 'DISK_DRIVE')
        # Topologically Transforming multiple object to all have the same topology by transforming one reference object into everything else
        row = layout.row(align=True)
        row.operator('heart.object_transform', text = "Transform Objects", icon = "SHAPEKEY_DATA")

#------

#BUTTONS


#----
class MESH_OT_import_ventricle(bpy.types.Operator):
    """Imports all stl files contained within the import directory"""
    bl_idname = 'heart.import_ventricle'
    bl_label = 'Import ventricle data'

    def execute(self,context):
        scene = context.scene
        
        import_dir_raw = bpy.path.abspath((scene.ventricle_import_dir or "").strip())
        if not import_dir_raw:
            import_dir_raw = "//"
        for name in os.listdir(import_dir_raw):
            sub = re.compile(r'_\d+')
            if sub.search(name): bpy.ops.import_mesh.stl(filepath=os.path.join(import_dir_raw,name))
        return {'FINISHED'}
    
class MESH_OT_quick_reset(bpy.types.Operator):
    """Quickly resets the state and reads the files in the temp folder in the import directory (WILL DELETE ANY SELECTED VERTICES)"""
    bl_idname = 'heart.quick_reset'
    bl_label = 'Quick reset'

    def execute(self,context):
        scene = context.scene
        view_layer = context.view_layer
        
        import_dir_raw = bpy.path.abspath((scene.ventricle_import_dir or "").strip())
        if not os.path.isdir(os.path.join(import_dir_raw,'rotated')): 
            cons_print("No temporary savestate found.")
            return {'CANCELLED'}

        # ------------------------------------------------------------------
        # 1) Wipe the selected vertices
        # ------------------------------------------------------------------
        # --- Remember original selection & active object, also which meshes were selected ---
        original_selection = list(context.selected_objects)
        original_active = view_layer.objects.active
        selected_meshes_for_stl = [obj for obj in original_selection if obj.type == 'MESH']

        def restore_selection():
            bpy.ops.object.select_all(action='DESELECT')

            # Reselect original objects (only if they still exist)
            for obj in original_selection:
                if obj and obj.name in view_layer.objects:
                    view_layer.objects[obj.name].select_set(True)

            # Restore active object if it still exists
            if original_active and original_active.name in view_layer.objects:
                view_layer.objects.active = view_layer.objects[original_active.name]
        ventricles = find_ventricle_objects(original_selection)
        for obj in ventricles:    
            obj.select_set(True)
        bpy.ops.object.delete()

        # ------------------------------------------------------------------
        # 2) Reimport everything
        # ------------------------------------------------------------------
        # Reimports after wiping the ventricles
        if not import_dir_raw:
            import_dir_raw = "//"
        temp_path = os.path.join(import_dir_raw,'rotated')
        for name in os.listdir(temp_path):
            if not name=="Connectivity" and not name == "ventricle_export_manifest.txt": bpy.ops.import_mesh.stl(filepath=os.path.join(temp_path,name))
        return {'FINISHED'}

class MESH_OT_export_ventricle(bpy.types.Operator):
    """Exports ventricle coordinates, connectivity and STL"""
    bl_idname = 'heart.export_ventricle'
    bl_label = 'Export ventricle'

    def execute(self, context):
        scene = context.scene
        view_layer = context.view_layer

        # --- Remember original selection & active object, also which meshes were selected ---
        original_selection = list(context.selected_objects)
        original_active = view_layer.objects.active
        selected_meshes_for_stl = [obj for obj in original_selection if obj.type == 'MESH']

        def restore_selection():
            bpy.ops.object.select_all(action='DESELECT')

            # Reselect original objects (only if they still exist)
            for obj in original_selection:
                if obj and obj.name in view_layer.objects:
                    view_layer.objects[obj.name].select_set(True)

            # Restore active object if it still exists
            if original_active and original_active.name in view_layer.objects:
                view_layer.objects.active = view_layer.objects[original_active.name]

        # ------------------------------------------------------------------
        # 1) Find ventricles only among the currently selected objects
        # ------------------------------------------------------------------
        ventricles = find_ventricle_objects(original_selection)
        if not ventricles:
            cons_print("Export ventricle: no ventricle_* meshes found in the current selection.")
            return {'CANCELLED'}

        # ------------------------------------------------------------------
        # 2) Node-connectivity check using only these ventricles
        # ------------------------------------------------------------------
        bpy.ops.object.select_all(action='DESELECT')
        for obj in ventricles:
            obj.select_set(True)
        view_layer.objects.active = ventricles[0]

        # This prints success message itself if everything is fine
        if not check_node_connectivity(context):
            restore_selection()
            return {'CANCELLED'}

        # ------------------------------------------------------------------
        # 3) Determine export directories:
        #    base_dir (for STL + manifest) + Connectivity subfolder
        # ------------------------------------------------------------------
        export_dir_raw = bpy.path.abspath((scene.ventricle_export_dir or "").strip())
        if not export_dir_raw:
            export_dir_raw = "//"

        base_dir = os.path.normpath(bpy.path.abspath(export_dir_raw))
        connectivity_dir = os.path.normpath(os.path.join(base_dir, "Connectivity"))

        try:
            os.makedirs(connectivity_dir, exist_ok=True)
        except Exception as e:
            cons_print(f"Export ventricle: cannot create directory '{connectivity_dir}': {e}")
            restore_selection()
            return {'CANCELLED'}

        # ------------------------------------------------------------------
        # 4) Export connectivity (faces) from the first ventricle
        # ------------------------------------------------------------------
        ref_obj = ventricles[0]
        mesh = ref_obj.data

        # Ensure mesh is triangulated (for connectivity consistency)
        for poly in mesh.polygons:
            if len(poly.vertices) != 3:
                cons_print(
                    f"Export ventricle: non-triangular face {poly.index} in "
                    f"'{ref_obj.name}'. Please triangulate the mesh first."
                )
                restore_selection()
                return {'CANCELLED'}

        faces_path = os.path.join(connectivity_dir, "ventricle_faces.txt")
        try:
            with open(faces_path, "w") as f:
                for poly in mesh.polygons:
                    v0, v1, v2 = poly.vertices
                    f.write(f"{v0} {v1} {v2}\n")
        except Exception as e:
            cons_print(f"Export ventricle: error writing faces file: {e}")
            restore_selection()
            return {'CANCELLED'}

        # ------------------------------------------------------------------
        # 5) Export per-frame vertex coordinates for ventricles
        #    ALWAYS named ventricle_verts_0, ventricle_verts_1, ...
        #    Also collect data for the manifest.
        # ------------------------------------------------------------------
        manifest_entries = []  # (index, stl_file, verts_file, source_name)

        for export_index, obj in enumerate(ventricles):
            verts_filename = f"ventricle_verts_{export_index}.txt"
            verts_path = os.path.join(connectivity_dir, verts_filename)

            try:
                with open(verts_path, "w") as f:
                    for v in obj.data.vertices:
                        co_world = obj.matrix_world @ v.co
                        f.write(f"{co_world.x:.8f} {co_world.y:.8f} {co_world.z:.8f}\n")
            except Exception as e:
                cons_print(
                    f"Export ventricle: error writing vertices for '{obj.name}': {e}"
                )
                restore_selection()
                return {'CANCELLED'}

            stl_filename = f"ventricle_{export_index}.stl"
            manifest_entries.append(
                (export_index, stl_filename, os.path.join("Connectivity", verts_filename), obj.name)
            )

        # ------------------------------------------------------------------
        # 6) Restore original selection (for user convenience)
        # ------------------------------------------------------------------
        restore_selection()

        # ------------------------------------------------------------------
        # 7) Export STLs
        #       - ventricles -> ventricle_0.stl, ventricle_1.stl, ...
        #       - all other originally selected meshes -> <safe_name>.stl
        # ------------------------------------------------------------------
        depsgraph = context.evaluated_depsgraph_get()

        # 7a) ventricles with canonical names
        for export_index, obj in enumerate(ventricles):
            stl_path = os.path.join(base_dir, f"ventricle_{export_index}.stl")
            export_object_to_stl(obj, stl_path, depsgraph=depsgraph)

        # 7b) other selected meshes (non-ventricles) with their own names
        other_meshes = [obj for obj in selected_meshes_for_stl if obj not in ventricles]
        for obj in other_meshes:
            safe_name = re.sub(r'[^0-9A-Za-z_]+', '_', obj.name)
            stl_path = os.path.join(base_dir, f"{safe_name}.stl")
            export_object_to_stl(obj, stl_path, depsgraph=depsgraph)

        # ------------------------------------------------------------------
        # 8) Write manifest file mapping ventricle indices to files + original names
        # ------------------------------------------------------------------
        manifest_path = os.path.join(base_dir, "ventricle_export_manifest.txt")
        try:
            with open(manifest_path, "w") as mf:
                mf.write("# Ventricle export manifest\n")
                mf.write("# base directory for STL and this manifest: {}\n".format(base_dir))
                mf.write("# Connectivity files are in subfolder: {}\n".format(connectivity_dir))
                mf.write("#\n")
                mf.write("# index  STL_file            verts_file                         source_object_name\n")
                for idx, stl_file, verts_file, source_name in manifest_entries:
                    mf.write(
                        f"{idx:3d}  {stl_file:18s}  {verts_file:30s}  {source_name}\n"
                    )
        except Exception as e:
            cons_print(f"Export ventricle: error writing manifest file: {e}")
            # Don't cancel export; STLs and verts already written.

        # Single, light console summary line
        cons_print(
            f"Export ventricle: STL -> '{base_dir}', connectivity -> '{connectivity_dir}', manifest -> '{manifest_path}'"
        )

        return {'FINISHED'}

class MESH_OT_plot_STL(bpy.types.Operator):
    """Display the plot"""
    bl_idname = 'heart.plot_stl'
    bl_label = 'plot STL files'
    
    def execute(self,context):
        scene = context.scene
        
        plot_input_dir = bpy.path.abspath((scene.plot_input_path or "").strip())
        plot_base_dir = os.path.join(bpy.path.abspath((scene.ventricle_import_dir or "").strip()), "rotated")
        
        for filename in os.listdir(plot_input_dir):
            print(filename)
            if Path(filename).suffix == ".txt":
                inputsetting = os.path.join(plot_input_dir,filename)
                break
        
        (
            correlationFlag,
            frameID,
            sFrameID,
            eFrameID,
            numFrame,
            numInter,
            interpolMethod,
        ) = parseRunTimeVariables_unique(inputsetting)

        # Pre-check STL and connectivity inputs
        check_stl_and_connectivity(plot_input_dir, numFrame, sFrameID, eFrameID)

        _,_,_, fig = derive_ed_es_from_volume_curve(inputsetting, plot_input_dir, plot_base_dir, interpolMethod, "volume", False)

        plt.show(block=True)
        
        return{'FINISHED'}   
     
class MESH_OT_save_plot_STL(bpy.types.Operator):
    """Save the plot without displaying and also any additional files within the input folder"""
    bl_idname = 'heart.save_plot_stl'
    bl_label = 'plot STL files'
    
    def execute(self,context):
        scene = context.scene
        
        plot_input_dir = bpy.path.abspath((scene.plot_input_path or "").strip())
        plot_base_dir = os.path.join(bpy.path.abspath((scene.ventricle_import_dir or "").strip()), "rotated")
        
        for filename in os.listdir(plot_input_dir):
            print(filename)
            if Path(filename).suffix == ".txt":
                inputsetting = os.path.join(plot_input_dir,filename)
                break
        
        (
            correlationFlag,
            frameID,
            sFrameID,
            eFrameID,
            numFrame,
            numInter,
            interpolMethod,
        ) = parseRunTimeVariables_unique(inputsetting)

        # Pre-check STL and connectivity inputs
        check_stl_and_connectivity(plot_input_dir, numFrame, sFrameID, eFrameID)
        
        os.makedirs(os.path.join(plot_input_dir,"volume_comparison"), exist_ok=True)

        _,_,_, fig = derive_ed_es_from_volume_curve(inputsetting, plot_input_dir, plot_base_dir, interpolMethod, "volume", True)

        plot_path = os.path.join(plot_input_dir,"volume_comparison","volume_curve_comparison.png")
        cons_print(plot_path)
        fig.savefig(plot_path)
        
        return{'FINISHED'}

class MESH_OT_calculate_valve_diameter(bpy.types.Operator):
    """Calculates Valve Diameter"""
    bl_idname = 'heart.calculate_valve_diameter'
    bl_label = 'Calculate Valve Diameter'
    
    def execute(self,context):
        scene = context.scene
        diam_dir_raw = bpy.path.abspath((scene.ventricle_import_dir or "").strip())
        if not diam_dir_raw:
            diam_dir_raw = "//"
        cons_print(f"Currently calculating valve diameter")
        
        figs, res = main_calc_diameter(diam_dir_raw, 400)
        
        context.scene.mitral_radius_long = res[0] * 1000
        context.scene.mitral_radius_small = res[1] * 1000
        context.scene.aortic_radius = res[2] * 1000
        
        
        return{"FINISHED"}
            
# -------------------------------------------------------------------
# Helpers for ventricle detection and STL export
# -------------------------------------------------------------------

# Look for names ending in "_<digits>", e.g. "..._0", "..._00", "..._123"
VENTRICLE_SUFFIX_RE = re.compile(r"_(\d+)$")

def get_ventricle_index_from_name(name: str):
    """
    Return integer index from names like 'ventricle_0', 'LV_00', 'foo_000',
    or None if there is no '_<digits>' suffix.
    """
    m = VENTRICLE_SUFFIX_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None

def find_ventricle_objects(objs):
    """
    Return list of mesh objects in `objs` whose name ends with '_<digits>',
    sorted by that numeric suffix.
    """
    items = []
    for obj in objs:
        if obj.type != 'MESH':
            continue

        idx = get_ventricle_index_from_name(obj.name)
        if idx is None:
            continue

        items.append((idx, obj))

    items.sort(key=lambda x: x[0])
    return [o for (_, o) in items]

def export_object_to_stl(obj, filepath, depsgraph=None):
    """Custom ASCII STL export for a single object."""
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()

    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    mesh.calc_loop_triangles()

    mw = eval_obj.matrix_world
    nmat = mw.to_3x3()

    safe_name = re.sub(r'[^0-9A-Za-z_]+', '_', obj.name)

    with open(filepath, "w") as f:
        f.write(f"solid {safe_name}\n")
        for tri in mesh.loop_triangles:
            n = nmat @ tri.normal
            try:
                n.normalize()
            except Exception:
                pass
            f.write(f"  facet normal {n.x} {n.y} {n.z}\n")
            f.write("    outer loop\n")
            for li in tri.loops:
                vid = mesh.loops[li].vertex_index
                co = mw @ mesh.vertices[vid].co
                f.write(f"      vertex {co.x} {co.y} {co.z}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid\n")

    eval_obj.to_mesh_clear()

class MESH_OT_object_transform(bpy.types.Operator):
    """Reshape mesh object to preserve mesh connetivity and topology between objects. Currently uses the object with max mesh count as reference (WIP)"""
    bl_idname = 'heart.object_transform'
    bl_label = 'Object Transformation'
    def execute(self,context):
        scene = context.scene
        view_layer = context.view_layer

        # --- Remember original selection & active object, also which meshes were selected ---
        selected_objects = context.selected_objects
        
        if len(selected_objects) == 0 or not selected_objects:
            cons_print(f"No objects selected")
            return False

        # ---  Picks a reference object as a basis to be transformed  ---
        max = 0
        for a in selected_objects:
            if len(bmesh.from_edit_mesh(a.data).verts) > max:
                source = a
                max = len(bmesh.from_edit_mesh(a.data).verts)
        
        cons_print(f"Object used as reference {source.name}")
        
        bpy.ops.object.mode_set(mode='OBJECT') 
        
        for obj in selected_objects:
            if obj != source:
                cons_print(f"current object target {obj.name}")
                copied_source = copy_object(source.name, obj.name + "_transformed_CPD")
                CPD_transform(copied_source,obj)
                copied_source.data.update()
                copied_source_bvh = copy_object(source.name, obj.name + "_transformed_BVH")
                BvH_transform(copied_source_bvh,obj)
                copied_source_bvh.data.update()
        cons_print(f"Operation completed")    
        return{"FINISHED"}

def CPD_transform(source,target):
    source_vert = []
    for v in source.data.vertices:
        source_vert.append(list(v.co[:]))

    target_vert = []
    for v in target.data.vertices:
        target_vert.append(list(v.co[:]))

    reg = RigidRegistration(X=np.asarray(target_vert), Y=np.asarray(source_vert))
    TY, _ = reg.register()
    for i in range(len(source.data.vertices)):
        source.data.vertices[i].co = mathutils.Vector(TY[i])
    source.data.update()

def BvH_transform(source,target):
    bvh = BVHTree.FromObject(target, bpy.context.evaluated_depsgraph_get())
    
    for v in source.data.vertices:
        world_co = source.matrix_world @ v.co

        loc, _, _, _ = bvh.find_nearest(world_co)

        if loc:
            v.co = source.matrix_world.inverted() @ loc
    cons_print(f"Transformation from {source} to {target} successful")

classes = [
    PANEL_Files, MESH_OT_export_ventricle, MESH_OT_import_ventricle, PANEL_Position_Ventricle, MESH_OT_quick_reset, MESH_OT_ApproachSelection,
    PANEL_Valves, PANEL_Pipeline, PANEL_Setup_Variables, MESH_OT_get_node, MESH_OT_ventricle_rotate, MESH_OT_build_valves, MESH_OT_support_struct, 
    MESH_OT_Ventricle_Sort, MESH_OT_Quick_Recon, MESH_OT_remove_basal,
    MESH_OT_create_basal, MESH_OT_connect_apical_and_basal, MESH_OT_Ventricle_Interpolation, MESH_OT_Add_Vessels_Valves, MESH_OT_check_node_connectivity,
]

dev_classes = [PANEL_Poisson, MESH_OT_poisson, MESH_OT_create_valve_orifice, MESH_OT_connect_valves, PANEL_Dev_tools, MESH_DEV_volumes, MESH_DEV_indices, MESH_DEV_edge_index, MESH_DEV_color_min_dist, MESH_DEV_test, MESH_OT_plot_STL, MESH_OT_save_plot_STL, MESH_OT_object_transform, MESH_OT_calculate_valve_diameter]
  
def register():
    # Position variables.
    bpy.types.Scene.pos_top = bpy.props.FloatVectorProperty(name="Top position", default = (0,0,1))
    bpy.types.Scene.pos_bot = bpy.props.FloatVectorProperty(name="Top position", default = (0,0,0))
    bpy.types.Scene.pos_septum = bpy.props.FloatVectorProperty(name="Top position", default = (0,1,0))
    # Mitral valve.
    bpy.types.Scene.mitral_radius_long = bpy.props.FloatProperty(name="mitral_radius_long", default=6,  min = 0.01)
    bpy.types.Scene.mitral_radius_small = bpy.props.FloatProperty(name="mitral_radius_small", default=3,  min = 0.01)
    bpy.types.Scene.translation_mitral = bpy.props.FloatVectorProperty(name="Aortic valve translation", default = (0,0,1))
    bpy.types.Scene.angle_mitral = bpy.props.FloatVectorProperty(name="Aortic valve rotation", default = (0,0,0))
    # Aortic valve.
    bpy.types.Scene.aortic_radius = bpy.props.FloatProperty(name="aortic_radius", default=2,  min = 0.01)
    bpy.types.Scene.translation_aortic = bpy.props.FloatVectorProperty(name="Aortic valve translation", default = (0,0,1))
    bpy.types.Scene.angle_aortic = bpy.props.FloatVectorProperty(name="Aortic valve rotation", default = (0,0,0))
    # Support structure.
    bpy.types.Scene.ref_minima = bpy.props.FloatVectorProperty(name="Minima of reference object", default = (0,0,0))
    bpy.types.Scene.ref_maxima = bpy.props.FloatVectorProperty(name="Maxima of reference object", default = (0,0,1))
    # Cutting plane variables.
    bpy.types.Scene.remove_basal_threshold = bpy.props.FloatProperty(name="Threshold for the removal of the basal region", default=28.5,  min = 0)
    bpy.types.Scene.height_plane = bpy.props.FloatProperty(name="Cut-off value for the creation of the reference basal region", default=40,  min = 0.01)
    bpy.types.Scene.min_valves = bpy.props.FloatProperty(name="Minimal z-value of valves", default=45)
    # Approach selection.
    bpy.types.Scene.approach = bpy.props.IntProperty(name="Chosen modeling approach", default=3, min = 3, max = 5)
    bpy.types.Scene.mean_reference = bpy.props.BoolProperty(name="Mean volume as reference", default=True)
    # Possion algorithm.
    bpy.types.Scene.poisson_depth = bpy.props.IntProperty(name="Depth of possion algorithm", default=10,  min = 1)
    # Interpolation variables.
    bpy.types.Scene.time_rr = bpy.props.FloatProperty(name="Time RR-duration", default=0.6,  min = 0.01)
    bpy.types.Scene.time_diastole = bpy.props.FloatProperty(name="Time diastole", default=0.35,  min = 0.01) 
    bpy.types.Scene.frames_ventricle = bpy.props.IntProperty(name="Amount of frames ventricle after interpolation", default=10,  min = 10)
    # Connection algorithm variables.
    bpy.types.Scene.reference_object_name = bpy.props.StringProperty(name="Name of the reference object", default = "ventricle_0")
    bpy.types.Scene.inset_faces_refinement_steps = bpy.props.IntProperty(name="Refinement steps when insetting faces in the connection algorithm", default=1, min=1)
    bpy.types.Scene.connection_twist = bpy.props.IntProperty(name="Twist for bridging algorithm in connection", default=0)
    bpy.types.Scene.max_con_sm_iter = bpy.props.IntProperty(name="Maximum smoothing iterations for the smoothing of the connection between basal and apical region", default=25, min = 5)
    bpy.types.Scene.min_con_sm_iter = bpy.props.IntProperty(name="Minimum smoothing iterations for the smoothing of the connection between basal and apical region", default=2, min = 0)
    bpy.types.Scene.sm_reps = bpy.props.IntProperty(name="Repitions of reselection and smoothing application when smoothing basal and apical region", default=3, min = 0)

    # Import variables.
    bpy.types.Scene.ventricle_import_dir = bpy.props.StringProperty(
        name="Import folder",
        description="Folder to import the ventricle STL dataset",
        subtype='DIR_PATH',
        default="//",
    )

    # Export / CFD pipeline variables.
    bpy.types.Scene.ventricle_export_dir = bpy.props.StringProperty(
        name="Export folder",
        description="Folder to export for ventricle connectivity/coordinates and STL export",
        subtype='DIR_PATH',
        default="//",
    )
    
    bpy.types.Scene.plot_input_path = bpy.props.StringProperty(
        name = "Input directory for plotting",
        description="Directory for plotting input",
        subtype="DIR_PATH",
        default="//",
    )
    
    # Register UI-classes for Panels and functions.
    for c in classes: bpy.utils.register_class(c)
    # Register Development UI-classes.
    if dev_env_tools: 
        for c in dev_classes: bpy.utils.register_class(c)

    
def unregister(): # Unregister classes.
    for c in classes: bpy.utils.unregister_class(c)
    if dev_env_tools:
        for c in dev_classes: bpy.utils.unregister_class(c)

        
if __name__ == '__main__': 
    register() 