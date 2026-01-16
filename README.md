GPU_ID=0 ROBOT=piper CONFIG=custom_piper SAVE_ROOT=./datasets bash batch_collect_data.sh

bash test_one_task_one_robot.sh --gpu 0 --robot franka-panda --task beat_block_hammer


### aloha 相机位置

1. 默认

```xml



```


### franka 相机位置

1. 
```xml
  <link name="camera_base">
    <visual>
      <origin rpy="-1.57079632679 -1.57079632679 -1.57079632679" xyz="0.008 -0.019 0"/>
      <geometry>
        <mesh filename="franka_description/meshes/visual/camera_base.glb" />
      </geometry>
    </visual>
  </link>
  <joint name="hand_to_camera_mount" type="fixed">
    <parent link="panda_hand"/>
    <child link="camera_base"/>
    <origin rpy="3.14159265359 -1.57079632679 -1.57079632679" xyz="-0.01 -0.025 -0.025"/>
    <!-- <axis xyz="0 -1 0"/> -->
  </joint>

  <link name="camera">
    <visual>
      <origin rpy="1.57079632679 -0 1.57079632679" xyz="0.0 -0.033 -0.0"/>
      <geometry>
        <mesh filename="franka_description/meshes/visual/d435.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="camera_joint" type="fixed">
    <parent link="camera_base"/>
    <child link="camera"/>
    <origin rpy="1.57079632679 0 -0" xyz="0.042 -0.045 0.0"/>
```



2. 
```xml
  <link name="camera_base">
    <visual>
      <origin rpy="-1.57079632679 -1.57079632679 -1.57079632679" xyz="0.008 -0.019 0"/>
      <geometry>
        <mesh filename="franka_description/meshes/visual/camera_base.glb" />
      </geometry>
    </visual>
  </link>
  <joint name="hand_to_camera_mount" type="fixed">
    <parent link="panda_hand"/>
    <child link="camera_base"/>
    <origin rpy="3.14159265359 -1.57079632679 -1.57079632679" xyz="-0.01 -0.025 -0.025"/>
    <!-- <axis xyz="0 -1 0"/> -->
  </joint>

  <link name="camera">
    <visual>
      <origin rpy="1.57079632679 -0 1.57079632679" xyz="0.0 -0.033 -0.0"/>
      <geometry>
        <mesh filename="franka_description/meshes/visual/d435.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="camera_joint" type="fixed">
    <parent link="camera_base"/>
    <child link="camera"/>
    <origin rpy="1.57079632679 0 0.3" xyz="0.042 -0.055 0.0"/>
```

3. 

```urdf
  <link name="camera_base">
    <visual>
      <origin rpy="-1.57079632679 -1.57079632679 -1.57079632679" xyz="0.008 -0.019 0"/>
      <geometry>
        <mesh filename="franka_description/meshes/visual/camera_base.glb" />
      </geometry>
    </visual>
  </link>
  <joint name="hand_to_camera_mount" type="fixed">
    <parent link="panda_hand"/>
    <child link="camera_base"/>
    <origin rpy="3.14159265359 -1.57079632679 -1.57079632679" xyz="-0.01 -0.025 -0.025"/>
    <!-- <axis xyz="0 -1 0"/> -->
  </joint>

  <link name="camera">
    <visual>
      <origin rpy="1.57079632679 -0 1.57079632679" xyz="0.0 -0.033 -0.0"/>
      <geometry>
        <mesh filename="franka_description/meshes/visual/d435.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="camera_joint" type="fixed">
    <parent link="camera_base"/>
    <child link="camera"/>
    <origin rpy="1.57079632679 0 0.45" xyz="0.042 -0.065 0.0"/>
```



### ARX-X5 相机位置

1. 默认

```urdf
  <link name="camera_base">
    <visual>
      <origin xyz="0 0 0.0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/camera_base.glb" />
      </geometry>
    </visual>
  </link>
  <joint name="hand_to_camera_mount" type="fixed">
    <parent link="link6"/>
    <child link="camera_base"/>
    <origin xyz="0.057 0 0" rpy="0 0 -3.1416"/>
  </joint>

  <link name="camera">
    <visual>
      <origin rpy="0.0 0.0 -1.5707963267948966" xyz="0.0 0.0 -0.0"/>
      <geometry>
        <mesh filename="meshes/camera.glb" />
      </geometry>
    </visual>
  </link>
  <joint name="camera_joint" type="fixed">
    <parent link="camera_base"/>
    <child link="camera"/>
    <origin xyz="-0.0275 0.0 0.05" rpy="0 0.3491 3.141592653589793"/>
  </joint> 

```

2. 

```urdf
  <link name="camera_base">
    <visual>
      <origin xyz="0 0 0.0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/camera_base.glb" />
      </geometry>
    </visual>
  </link>
  <joint name="hand_to_camera_mount" type="fixed">
    <parent link="link6"/>
    <child link="camera_base"/>
    <origin xyz="0.01 0 0.01" rpy="0 0 -3.1416"/>
  </joint>

  <link name="camera">
    <visual>
      <origin rpy="0.0 0.0 -1.5707963267948966" xyz="0.0 0.0 -0.0"/>
      <geometry>
        <mesh filename="meshes/camera.glb" />
      </geometry>
    </visual>
  </link>
  <joint name="camera_joint" type="fixed">
    <parent link="camera_base"/>
    <child link="camera"/>
    <origin xyz="-0.0275 0.0 0.05" rpy="0 0.3491 3.141592653589793"/>
  </joint> 
```

3. 

```urdf
  <link name="camera_base">
    <visual>
      <origin xyz="0 0 0.0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/camera_base.glb" />
      </geometry>
    </visual>
  </link>
  <joint name="hand_to_camera_mount" type="fixed">
    <parent link="link6"/>
    <child link="camera_base"/>
    <origin xyz="-0.00 0 0.04" rpy="0 -0.15 -3.1416"/>
  </joint>

  <link name="camera">
    <visual>
      <origin rpy="0.0 0.0 -1.5707963267948966" xyz="0.0 0.0 -0.0"/>
      <geometry>
        <mesh filename="meshes/camera.glb" />
      </geometry>
    </visual>
  </link>
  <joint name="camera_joint" type="fixed">
    <parent link="camera_base"/>
    <child link="camera"/>
    <origin xyz="-0.0275 0.0 0.05" rpy="0 0.3491 3.141592653589793"/>
  </joint> 

```


### piper 相机位置

1. 默认

```urdf
  <link name="camera_mount">
    <visual>
      <origin xyz="0 0 0.08" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/camera_link.glb"/>
      </geometry>
    </visual>
  </link>
  <joint name="link6_to_camera_mount" type="fixed">
    <parent link="link6"/>
    <child link="camera_mount"/>
    <origin xyz="0 0 -0.077" rpy="0 0 1.5708"/>
  </joint>
  <link name="camera">
    <visual>
      <origin xyz="0.0015 -0.0315 -0.00" rpy="0.1 0 -1.5707963267948966"/>
      <geometry>
        <mesh filename="meshes/camera.glb"/>
      </geometry>
    </visual>
  </link>
  <joint name="camera_joint" type="fixed">
    <parent link="camera_mount"/>
    <child link="camera"/>
    <origin xyz="0.032 0.055 0.115" rpy="0 -1.1116926535897932 -1.5707963267948966"/>
  </joint>

```


2. 

```urdf
  <link name="camera_mount">
    <visual>
      <origin xyz="0 0 0.08" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/camera_link.glb"/>
      </geometry>
    </visual>
  </link>
  <joint name="link6_to_camera_mount" type="fixed">
    <parent link="link6"/>
    <child link="camera_mount"/>
    <origin xyz="0 0 -0.077" rpy="0 0 1.5708"/>
  </joint>
  <link name="camera">
    <visual>
      <origin xyz="0.0015 -0.0315 -0.00" rpy="0.1 0 -1.5707963267948966"/>
      <geometry>
        <mesh filename="meshes/camera.glb"/>
      </geometry>
    </visual>
  </link>
  <joint name="camera_joint" type="fixed">
    <parent link="camera_mount"/>
    <child link="camera"/>
    <origin xyz="0.012 0.06 0.115" rpy="0 -1.1116926535897932 -1.5707963267948966"/>
  </joint>
```


3. 

```urdf
  <link name="camera_mount">
    <visual>
      <origin xyz="0 0 0.08" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/camera_link.glb"/>
      </geometry>
    </visual>
  </link>
  <joint name="link6_to_camera_mount" type="fixed">
    <parent link="link6"/>
    <child link="camera_mount"/>
    <origin xyz="0 0 -0.077" rpy="0 0 1.5708"/>
  </joint>
  <link name="camera">
    <visual>
      <origin xyz="0.0015 -0.0315 -0.00" rpy="0.1 0 -1.5707963267948966"/>
      <geometry>
        <mesh filename="meshes/camera.glb"/>
      </geometry>
    </visual>
  </link>
  <joint name="camera_joint" type="fixed">
    <parent link="camera_mount"/>
    <child link="camera"/>
    <origin xyz="0.002 0.06 0.115" rpy="0 -1.1116926535897932 -1.5707963267948966"/>
  </joint>

```


### ur5-wsg 相机位置

1. 默认

```urdf
  <link name="camera_base">
    <visual>
      <origin rpy="0 0 1.57079632679" xyz="0 -0.0 0"/>
      <geometry>
        <mesh filename="meshes/ur5/visual/camera_base_ur5.glb" />
      </geometry>
    </visual>
  </link>
  <joint name="hand_to_camera_mount" type="fixed">
    <parent link="wrist_3_link"/>
    <child link="camera_base"/>
    <origin rpy="3.14159265359 1.57079632679 1.57079632679" xyz="0.0 0.08 0"/>
  </joint>

```

2. 

```urdf
  <link name="camera_base">
    <visual>
      <origin rpy="0 0 1.57079632679" xyz="0 -0.0 0"/>
      <geometry>
        <mesh filename="meshes/ur5/visual/camera_base_ur5.glb" />
      </geometry>
    </visual>
  </link>
  <joint name="hand_to_camera_mount" type="fixed">
    <parent link="wrist_3_link"/>
    <child link="camera_base"/>
    <origin rpy="3.14159265359 2.2 1.57079632679" xyz="0.0 0.1 0.04"/>
  </joint>
```

3. 

```urdf
  <link name="camera_base">
    <visual>
      <origin rpy="0 0 1.57079632679" xyz="0 -0.0 0"/>
      <geometry>
        <mesh filename="meshes/ur5/visual/camera_base_ur5.glb" />
      </geometry>
    </visual>
  </link>
  <joint name="hand_to_camera_mount" type="fixed">
    <parent link="wrist_3_link"/>
    <child link="camera_base"/>
    <origin rpy="3.14159265359 2.0 1.57079632679" xyz="0.0 0.11 0.04"/>
  </joint>
```