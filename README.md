# mapless-navigation

## Standard Build
#### 1. Install dependencies:
```bash
apt install libpcl-dev liboctomap-dev
```

#### 2. Build:
```bash
mkdir build
cmake -S src/base_src -B build
make -C build
```

#### 3. Run tests:
```bash
./build/tests
```

#### 4. Create a Python environment:
```bash
virtualenv -p python3 ./venv
source ./venv/bin/activate
pip install numpy open3d matplotlib pandas
```

## ROS 2 Build
#### 1. Install `ROS2` and `colcon`

#### 2. Install dependencies
```bash
apt install ros-<version>-octomap
apt install ros-<version>-octomap-ros
```

#### 3. Create a Python environment:
```bash
virtualenv -p python3 ./venv
source ./venv/bin/activate
touch ./venv/COLCON_IGNORE 
pip install catkin_pkg rosbags lark empy==3.3.4
pip install numpy open3d matplotlib pandas
```

#### 4. Check OpenCV Path
In `src/CMakeLists.txt`, check the path to the library here
```text
set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
```

To list your paths and libraries, use
```bash
pkg-config --cflags opencv4
pkg-config --libs opencv4
```

#### 5. Build
```bash
colcon build
```
Source the workspace:
```bash
source install/setup.bash
```

#### 6. Run tests
```bash
colcon test
```
Verbose testing:
```bash
colcon test --event-handler console_direct+
```

## Dataset
Download from
```
https://arpg.github.io/coloradar
```
or
```
http://arpg-storage.cs.colorado.edu:8080/
```
Put `utils/calib` and `kitti` into `<dataset>/calib` and `<dataset>/kitti`. Suggested path is `<HOME>/coloradar`.

[//]: # (#### 3.1 Convert bags into ros2 bags)

[//]: # (```bash)

[//]: # (chmod +x src/scripts/convert_bags_to_ros2.sh)

[//]: # (./src/scripts/convert_bags_to_ros2.sh)

[//]: # (```)


## Usage
Open `run.ipynb` and follow the instructions.
