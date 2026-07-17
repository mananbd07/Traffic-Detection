import carla
import cv2
import numpy as np
import time

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("Digital Twin Feed", i3)
    cv2.waitKey(1)

# --- THE CONNECTION LOGIC ---
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(20.0) # 20 seconds is plenty

connected = False
while not connected:
    try:
        print("📡 Attempting to connect to CARLA Server...")
        world = client.get_world()
        connected = True
        print("✅ Connection Successful!")
    except Exception as e:
        print(f"⌛ Server not ready yet... retrying in 5s")
        time.sleep(5)

# --- SETTING UP THE TWIN ---
blueprint_library = world.get_blueprint_library()
cam_bp = blueprint_library.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '800')
cam_bp.set_attribute('image_size_y', '600')

# Position the camera at a busy intersection
spawn_point = carla.Transform(carla.Location(x=-10, y=130, z=20), carla.Rotation(pitch=-35))
sensor = world.spawn_actor(cam_bp, spawn_point)

sensor.listen(lambda data: process_img(data))

try:
    while True:
        world.wait_for_tick()
except KeyboardInterrupt:
    sensor.destroy()
    cv2.destroyAllWindows()
    print("Cleaned up.")