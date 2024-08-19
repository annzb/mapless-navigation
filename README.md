# mapless-navigation

## Requirements
#### 1. Install `ROS2` and `colcon`

#### 2. Install dependencies
```bash
apt install ros-<version>-octomap
apt install ros-<version>-octomap-ros
```

#### 2.1 Optional Python venv
Create virtual environment:
```bash
virtualenv -p python3 ./venv
source ./venv/bin/activate
touch ./venv/COLCON_IGNORE
```

Install Python packages:
```bash
(.venv) pip install catkin_pkg
```

#### 2.2 OpenCV path
In `CMakeLists.txt`, change the path to the library here
```text
set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
```

To list your paths and libraries, use
```bash
pkg-config --cflags opencv4
pkg-config --libs opencv4
```

## Dataset
```
https://arpg.github.io/coloradar
http://arpg-storage.cs.colorado.edu:8080/
```

Put `calib`, `bags`, `kitti` into `.../coloradar`.

## Usage
Build and test package:
```bash
source venv/bin/activate
colcon build
colcon test
```
Verbose testing:
```bash
colcon test --event-handler console_direct+
```

