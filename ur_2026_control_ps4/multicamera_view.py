import pyrealsense2 as rs
import numpy as np
import cv2

ctx = rs.context()
devices = ctx.query_devices()
num_devices = len(devices)

if num_devices == 0:
    print("No RealSense devices connected.")
    exit(0)

print(f"Found {num_devices} device(s)")

pipelines = []
for i, dev in enumerate(devices):
    serial = dev.get_info(rs.camera_info.serial_number)
    product_line = str(dev.get_info(rs.camera_info.product_line))
    print(f"  Device {i}: {dev.get_info(rs.camera_info.name)} (S/N: {serial})")

    pipeline = rs.pipeline(ctx)
    config = rs.config()
    config.enable_device(serial)

    if product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    pipelines.append((pipeline, serial))

try:
    while True:
        for i, (pipeline, serial) in enumerate(pipelines):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            win_name = f"Camera {i} ({serial})"
            cv2.imshow(win_name, color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    for pipeline, _ in pipelines:
        pipeline.stop()
    cv2.destroyAllWindows()