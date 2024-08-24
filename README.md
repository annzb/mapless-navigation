# mapless-navigation

## Requirements

[//]: # (### 1. Install `ROS2` and `colcon`)

### 1. Install dependencies
```bash
apt install ros-<version>-octomap
apt install ros-<version>-octomap-ros
```

#### 1.1 Optional Python venv
Create virtual environment:
```bash
virtualenv -p python3 ./venv
source ./venv/bin/activate
touch ./venv/COLCON_IGNORE  # if using ros2
```

#### 2.2 Python packages
```bash
pip install numpy open3d matplotlib pandas
```
If using `ros2`, install:
```bash
pip install catkin_pkg rosbags lark empy==3.3.4
```

#### 2.3 OpenCV path with ros2
In `CMakeLists.txt`, change the path to the library here
```text
set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
```

To list your paths and libraries, use
```bash
pkg-config --cflags opencv4
pkg-config --libs opencv4
```

### 2. Get dataset
Download from
```
https://arpg.github.io/coloradar
```
or
```
http://arpg-storage.cs.colorado.edu:8080/
```
Put `utils/calib`, `kitti` into `<HOME>/coloradar`.

[//]: # (#### 3.1 Convert bags into ros2 bags)

[//]: # (```bash)

[//]: # (chmod +x src/scripts/convert_bags_to_ros2.sh)

[//]: # (./src/scripts/convert_bags_to_ros2.sh)

[//]: # (```)


## Usage
### 0. Build and test the package
Example using `ros2` + Python `venv`:
```bash
source venv/bin/activate
colcon build
colcon test
```
Verbose testing:
```bash
colcon test --event-handler console_direct+
```
Source the workspace:
```bash
source install/setup.bash
```

### 1. Open `run.ipynb` and follow the instructions
