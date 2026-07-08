# Geometric-ventricle-reconstruction Pipeline
## Description
Blender-addon for the geometric reconstruction of time-varying 3D ventricle geometries lacking good spatial resolution of the mitral and the aortic valve. Detailed information can be found in this [paper](https://doi.org/10.1002/eng2.13041).

Three accompanying videos exist and will be referred to throughout the tutorial.

Installation - https://www.youtube.com/watch?v=cKEKuLW4oYE

Use - https://www.youtube.com/watch?v=0sduwcDeSm8

Downstream CFD simulations - https://www.youtube.com/watch?v=C1O20YvkCJs

# Installation
Video walking through installation: https://www.youtube.com/watch?v=cKEKuLW4oYE
- Install Blender 3.1
- Install Python (tested with Python 3.12.2)
- Install pip: https://pip.pypa.io/en/stable/installation/
- Install Python Packages open3D and numba in console (using virtual environment recommended)
```bash
py -3.12 -m venv .venv-blender
python.exe -m pip install -U pip wheel setuptools
pip install open3d numba scipy
```
## Running Blender with Python environment variables
Windows: Run Powershell or any other terminal \
Go to the directory of Blender and run it with Python system environment
```bash
cd <BLENDERPATH>
./blender.exe --python-use-system-env
```
## Installation of scipy numba in Blender Python console
After opening Blender 3.1 using Powershell load the blend-file provided in the repository. It contains objects used in the addon.
Open the integrated Blender Python console.
```bash
import sys
import subprocess
subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'ensurepip'])
subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'pip', 'install', 'numba'])
subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'pip', 'install', 'scipy'])
subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'pip', 'install', 'open3d'])
subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'pip', 'install', 'matplotlib'])
subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'pip', 'install', 'PyQt5'])
subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'pip', 'install', 'trimesh'])
subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'pip', 'install', 'pandas'])
```
If pip is missing (Output 0 in Blender Python console):
```bash
import ensurepip
ensurepip.bootstrap()
from pip._internal import main
main(args=['install','numba'])
main(args=['install','scipy'])
main(args=['install','open3d'])
main(args=['install','matplotlib'])
main(args=['install','PyQt5'])
main(args=['install','trimesh'])
main(args=['install','pandas'])
```
## Installation of Blender case with addons
In Blender go to Edit→Preferences→Add-ons:
- Search *LoopTools* → check *Mesh: LoopTools* → Save Preferences.
- Install stl-plot.py, calculate_valve_diameters.py and then ventricle-reconstruction-pipeline.py (**in this order**) from repository → search *Geometrical heart* → check *Add Mesh: Geometrical heart reconstruction*
- Open the GVR-Pipeline.blend scene from the repository → after that a new category should appear on the right side of the 3D Viewport called 'GVR-Pipeline'. Clicking it will open panels containing buttons, etc. used for the pipeline.
# Usage
Video-tutorial: https://www.youtube.com/watch?v=0sduwcDeSm8 \
Installation description: https://www.youtube.com/watch?v=cKEKuLW4oYE \
→ Open blend-file provided in the repository and follow the following steps.
## Import ventricle geometries 
![Image of the import folder](/readme_images/Import.png)\
In the File Panel select an import folder either by manually input or by clicking the folder icon on the right side to browse through the file explorer. 
Afterwards just click "Import ventricles" to import all the *.STL* files within the selected folder automatically.
## Setup pipeline
![Image of the setup pipeline](/readme_images/Pipeline.png)\
The steps for using this pipeline are indicated by the buttons F1 up to F5. 
F1 and F5 includes a clickable button for sorting the volumes and selecting the approach respectively. 
1. Sort volumes\
    1.1. Select all volumes\
    1.2. Click button 'Sort volumes'\
    ![Image of the setup pipeline](/readme_images/Pipeline_button_one.png)
    \
    This restructures the list of selected objects such that the object with the smallest volume is the first object and all objects that were before that object are concatenated in the original order at the end of the object list. It also changes the names of the objects to the naming convention ventricle 0 ... X.
2. Setup ventricle position and rotation\
    2.1. Open panel 'Ventricle position (mm)'\
    ![Image of the Ventricle position panel](/readme_images/Ventricle_position.png)\
    2.2. Select only one ventricle and hide the others ('h'-key while selected)\
    2.3. Go into Edit mode\
    2.4. Select a single node centrally in the basal region and press button 'Select basal node' or manually change location of the basal node (before transformation) using the editboxes above the button\
    2.5. Select a single node at the ventricle apex and press button 'Select apex node' or manually change location of the apex node (before transformation) using the editboxes above the button\
    2.6. Select a single node at the ventricle septal ventricle wall and press button 'Select node at septum' or manually change location of the septal ventricle node (before transformation) using the editboxes above the button\
    2.7. Leave Edit mode\
    2.8. Unhide all ventricles and reselect them\
    2.9. Press button 'Translate and rotate'. This will rotate the ventricle and saves the result on a temporary save file named 'rotate' within the same directory as the import directory.\
    \
    Three points are selected on a ventricle to translate and rotate the ventricles. These transformations of the local ventricle coordinate system to a global coordinate system streamline the handling of the ventricle objects in future steps.
3. Setup valves\
    3.1. Open panel 'Valve options'\
    ![Image of the Valve options panel](/readme_images/Valve_options.png)\
    3.2. Change positition (translation), rotation (angle) and size (radii) of the mitral and aortic valve using the respective textboxes\
    \
    This sets up the arrangement of the mitral and aortic valve. (These inputs can be checked by pressing the buttons 'Add valve interface nodes' and 'Build support structure around valves'. Note that this will add nodes to an existing object. So consider creating a copy before pressing those buttons.) An additional button "Calculate Diameter" also reads directly from the import folder and immediately sets the value of the Long mitral radius, Small mitral radius, and Aortic Radius. 
4. Setup algorithm variables\
    4.1. Open panel 'Algorithm setup variables'\
    ![Image of the Algorithm setup variables panel](/readme_images/Algorithm_setup.png)\
    4.2. Change variables for the algorithm. Threshold needs to be adjusted depending on geometry. The other settings are advanced and should not be changed lightly.\
    \
    Description variables:
    - Threshold for basal region removal: Cartesian z-coordinate. All vertices above this threshold are deleted during the basal region removal
    - Use mean instead of max volume as reference: Changes the method for finding the reference ventricle to either the max or mean volume (True = Mean volume, False = Max volume)
    - Time RR-duration: Cardiac cycle duration
    - Time diastole: Diastole duration
    - Frames after interpolation: When using approach A5 the ventricle objects are interpolated to this amount of timeframes
    - Depth of poisson surface reconstruction algorithm: Maximum tree depth for Poisson surface reconstruction (https://hhoppe.com/poissonrecon.pdf)
    - Twist during connection algorithm: Value for 'twist'-variable used in the bridge function of the looptools addon used to connect the apical and the basal region. (Usally the default value 0 fits best)
    - Refinement steps for insetting faces: Amount of iterations of insetting faces during the connection algorithm
    - Maximum smoothing iterations: Used in smoothing the connection of basal and apical region. Highest (initial) smoothing value
    - Minimum smoothing iterations: Used in smoothing the connection of basal and apical region. Smallest smoothing value
    - Smoothing repetitions: Used in smoothing the connection of basal and apical region. Amount of smoothing repetitions each with a wider node selection (all neighbours of previous selection are selected)
5. Select approach\
    5.1. In panel 'Geometric ventricle reconstrucion pipeline press button 'Select approach'\
    ![Image of the setup pipeline](/readme_images/Pipeline_button_five.png)\
    5.2. In pop-up window choose approach from drop-down menu and confirm with 'OK'\
    \
    Change the valve modeling approach.
## Run pipeline
Select all ventricle objects and either run all steps with the button 'Quick reconstruction' in the panel 'Geometric ventricle reconstruction pipeline' or do the following steps for a more comprehensive execution of the pipeline:\
![Image of the setup pipeline](/readme_images/Pipeline_optional.png)
1. Remove basal region\
    1.1. Press button 'Remove basal region' in the panel 'Geometric ventricle reconstruction pipeline'\
    \
    This removes all vertices above the z-value for a reference ventricle. The vertices of the other ventricle object with identical indices to the deleted one in the reference are also deleted. The upper edge loop is smoothed such that all its vertices lay on the same xy-plane. Lastly the ventricle objects are shifted along the z-axis such that all xy-planes match with the reference ventricle xy-plane.
2. Create basal region\
    2.1. Press button 'Create basal region' in the panel 'Geometric ventricle reconstruction pipeline'\
    \
    This creates a reference basal region used the selected object. For that first the valve indices and a support structure are added to a copy of the reference ventricle. Then the Poisson surface reconstruction is applied to the vertices to create a surface object from all vertices. After that the object is remeshed and the apical region is removed while smoothing the lower edge loop of the resulting basal region. If multiple objects are selected, then a basal region will be created for each of the selected object, however, these basal regions will most likely not share the same topology. The button "Mesh Transformation" is required to remesh the objects to share the same topology.
3. Connect basal and apical parts\
    3.1. Press button 'Connect basal and apical regions' in the panel 'Geometric ventricle reconstruction pipeline'\
    \
    This creates a copy of the reference basal region for all apical region ventricle objects and connects them with the looptools_bridge function from the Blender addon Looptools. Since this connection creates long quadrangular faces, the faces need to be split using an integrated insetting algorithm leading to faces where the deviation of edge lengths are reduced. After that the faces are triangulated and iteratively smoothed. These processes are done for the reference ventricle object first and then copied to the other ventricles to remain node-connectivity.
4. Add atrium, aorta and valves\
    4.1. Press button 'Add atrium, aorta and valves' in the panel 'Geometric ventricle reconstruction pipeline'\
    \
    This copies objects for aorta, atrium, mitral and aortic valve found in the Blender-project and scales, rotates and positions them at their respective places.

## Quick reset
![Image of the dev tools panel](/readme_images/File_management_quick_reset.png)\
By selecting all objects to delete and pressing quick reset, it will import all the files saved from the last 'Translate and Rotate' process. However, it will still use the directory specified in the "Import folder" as a reference point to find the temporary export to quick import.

## Export files
Option 1: Export ventricle function
- specify the path in `Export folder`
- select all objects to export
- klick `Export ventricle`\
![Image of the dev tools panel](/readme_images/File_management_export_directory.png)

Option 2: Use Blender built-in export. For this:

Select all objects to export. Then go to `File→Export→.STL→...`
- tick ASCII checkbox
- Batch Mode Object
- tick selection only checkbox
- keep the other options at default
- leave name empty\
→export STL

## Optional usage: Development tools panel
![Image of the dev tools panel](/readme_images/Dev_tools.png)
### Compute volumes
Compute volumes of all selected objects and prints them to the Blender Python console.
### Get vertex indices
Print indices and their position vectors of all selected vertices.
### Get edge index
While exactly two neighbouring vertices are selected print the vertex indices and the edge index.
### Node-connectivity check
Check the node-connectivity of all selected objects. This includes:
- a check if there are any nodes with only two neighbouring vertices (this would lead to bad triangle face generation when exporting from blender)
- a check if the amount of vertices, edges and faces match
- a check if the edges and faces of all objects are created with the same vertices
### Color minimal distance to raw object
During the usage of the pipeline the longitudinal shift is saved as a variable bound to the respective object (ventricle 0 ... X). The user has to re-import the raw data and rename it to 'ref_obj'. While the reconstructed object is selected pressing the button 'Color minimal distance to raw object' will compute the minimal distance from each face-center of the reconstructed ventricle to any face-center of the reference object resulting in a 3d-representation of the Hausdorff distance (https://cgm.cs.mcgill.ca/~godfried/teaching/cg-projects/98/normand/main.html). The faces of the object are then colored with the distances which are normalized with the maximum value resulting in a scale from 0 to 1 (blue→white→red). To view the colors select 'Material Preview' in Blender (top right in 3D Viewport). This process give a qualitative visual representation of the quality of the reconstruction pipeline.

### Transform Objects
By selecting multiple meshes with similar orientation, this button picks a mesh to use as reference, and then creates of copy and transform it into each of the other selected meshes. This aims to make sure that all the meshes share the same topology. It is transformed using a BvH Tree based transformation.

# Comparison of volume curves (raw vs. reconstructed)
![Image of the plotting functionality](/readme_images/Plot_function.png)\
This panel (labelled **"Comparison of volume curves"** in the *Development tools* panel) compares the left-ventricular **volume over the cardiac cycle** of your **raw input data** against the **reconstructed geometries** produced by this pipeline. It draws three curves — the raw frames, the reconstructed frames, and the temporally interpolated reconstruction — and marks end-diastole (**ED**, volume maximum) and end-systole (**ES**, volume minimum). Use it to judge how well the reconstruction preserves the volume dynamics.

Volumes are computed from the mesh **connectivity** (`Connectivity/*.txt`), not from the STL files directly, so both the raw and the reconstructed side must have been exported together with their `Connectivity/` folder (see the structure below).

**Prerequisites**
- Run **"Translate and rotate"** first — it writes the raw, aligned data (including its `Connectivity/`) into `<Import folder>/rotated/`.
- **Export** the reconstructed ventricles (**"Export ventricle"**) so the processed geometries also have a `Connectivity/` folder.

**Inputs — set these before running**
- **Import folder** (in the *File* panel) — points at the raw STL frames. The raw comparison data is taken from `<Import folder>/rotated/`. If that folder is missing, you are asked to run *Translate and rotate* first.
- **`inputPython.txt`** — the settings file shared with the rest of the pipeline (RR duration, number of frames, start/end frame, interpolation method). It is found **automatically** in the *Import folder* or **one level above** it — there is no field for it.
- **Processed geometries** (in this panel) — the folder holding the reconstructed geometries to compare. **If it is left empty, the *Export folder* is used instead.**

**Running it**
- **Show** — displays the plot. If an interactive matplotlib backend is available (a Qt binding such as *PyQt5* installed in Blender's Python), an interactive window opens; otherwise the plot is written to a temporary PNG and opened with your operating system's default image viewer.
- **Save** — writes the plot and the underlying data to disk (see *Outputs*).

**Outputs** — **Save** writes into `<Processed geometries>/volume_comparison/`:
- `volume_curve_comparison.png` — the plotted curves
- `_volume_frames_volumes.csv` — per-frame volumes of the reconstructed geometry
- `_volume_interp_volumes.csv` — the interpolated volume curve

If a required input is missing (no `rotated/`, no `inputPython.txt`, or no valid *Processed geometries* folder), a clear warning is printed to the Blender **System Console** (`Window → Toggle System Console`) and nothing is plotted.

**Case folder structure**\
A typical case is organised as shown below; the entries relevant to this panel are annotated. The *Processed geometries* / *Export folder* does **not** have to live inside the case folder — it can be anywhere; this is just a common layout.
```
<case>/
├── inputPython.txt              ← settings; found here or inside the Import folder
├── STL/                         ← set as "Import folder" (the raw STL frames)
│   └── rotated/                 ← created by "Translate and rotate" -> RAW comparison data
│       ├── ventricle_0.stl, ventricle_1.stl, ...
│       └── Connectivity/
│           ├── ventricle_faces.txt
│           └── ventricle_verts_0.txt, ventricle_verts_1.txt, ...
├── export/                      ← "Export folder" / "Processed geometries" -> RECONSTRUCTED data
│   ├── ventricle_0.stl, ventricle_1.stl, ...
│   ├── Connectivity/
│   │   ├── ventricle_faces.txt
│   │   └── ventricle_verts_0.txt, ...
│   └── volume_comparison/       ← created by "Save" (png + csv outputs)
└── calc_valve_diameters_outputs/
```

# Authors and acknowledgment
- Author: Daniel Verhülsdonk
- Supervision by: Jan-Niklas Thiel (thiel@ame.rwth-aachen.de) and Michael Neidlin

# Application
Video on how to use the pipeline in further CFD simulations at https://www.youtube.com/watch?v=C1O20YvkCJs\
This tool was used in the following 2 publications:

### 1. Quantifying the Impact of Mitral Valve Anatomy on Clinical Markers Using Surrogate Models and Sensitivity Analysis
https://doi.org/10.1016/j.compbiomed.2025.110265

This pipeline was used to create the ventricle geometries that were used to run Ansys Fluent CFD simulations necessary for training the surrogate models. More details on using this automated CFD model and the corresponding setup files can be found here:

https://doi.org/10.5281/zenodo.12519189

https://www.youtube.com/watch?v=gO0ZYzpblLA

### 2. An interactive computational pipeline to investigate ventricular hemodynamics with real-time three-dimensional echocardiography and computational fluid dynamics
https://doi.org/10.1002/eng2.13041

This pipeline was used to perform geometry processing for CFD models of ventricular blood flow. We showcase its use on real-time three-dimensional echocardiography data of three patient datasets from two different clinical centers.

# License
MIT License

Copyright (c) [2024] [Institute of Applied Medical Engineering - Cardiovascular Engineering (AME-CVE) RWTH Aachen]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
