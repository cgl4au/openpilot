import numpy as np

from common.transformations.camera import eon_focal_length, \
        vp_from_ke, \
        get_view_frame_from_road_frame, \
        FULL_FRAME_SIZE

# segnet

SEGNET_SIZE = (512, 384)

segnet_frame_from_camera_frame = np.array([
  [float(SEGNET_SIZE[0])/FULL_FRAME_SIZE[0],    0.,          ],
  [     0.,          float(SEGNET_SIZE[1])/FULL_FRAME_SIZE[1]]])


# model

MODEL_INPUT_SIZE = (320, 160)
MODEL_YUV_SIZE = (MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1] * 3 // 2)
MODEL_CX = MODEL_INPUT_SIZE[0]/2.
MODEL_CY = 21.

model_zoom = 1.25
model_height = 1.22

# canonical model transform
model_intrinsics = np.array(
  [[ eon_focal_length / model_zoom,    0. ,  MODEL_CX],
   [   0. ,  eon_focal_length / model_zoom,  MODEL_CY],
   [   0. ,                            0. ,   1.]])


# MED model
MEDMODEL_INPUT_SIZE = (512, 256)
MEDMODEL_YUV_SIZE = (MEDMODEL_INPUT_SIZE[0], MEDMODEL_INPUT_SIZE[1] * 3 // 2)
MEDMODEL_CY = 47.6

medmodel_zoom = 1.
medmodel_intrinsics = np.array(
  [[ eon_focal_length / medmodel_zoom,    0. ,  0.5 * MEDMODEL_INPUT_SIZE[0]],
   [   0. ,  eon_focal_length / medmodel_zoom,  MEDMODEL_CY],
   [   0. ,                            0. ,   1.]])


# BIG model

BIGMODEL_INPUT_SIZE = (864, 288)
BIGMODEL_YUV_SIZE = (BIGMODEL_INPUT_SIZE[0], BIGMODEL_INPUT_SIZE[1] * 3 // 2)

bigmodel_zoom = 1.
bigmodel_intrinsics = np.array(
  [[ eon_focal_length / bigmodel_zoom,    0. , 0.5 * BIGMODEL_INPUT_SIZE[0]],
   [   0. ,  eon_focal_length / bigmodel_zoom,  0.2 * BIGMODEL_INPUT_SIZE[1]],
   [   0. ,                            0. ,   1.]])


bigmodel_border = np.array([
    [0,0,1],
    [BIGMODEL_INPUT_SIZE[0], 0, 1],
    [BIGMODEL_INPUT_SIZE[0], BIGMODEL_INPUT_SIZE[1], 1],
    [0, BIGMODEL_INPUT_SIZE[1], 1],
])


model_frame_from_road_frame = np.dot(model_intrinsics,
  get_view_frame_from_road_frame(0, 0, 0, model_height))

bigmodel_frame_from_road_frame = np.dot(bigmodel_intrinsics,
  get_view_frame_from_road_frame(0, 0, 0, model_height))

medmodel_frame_from_road_frame = np.dot(medmodel_intrinsics,
  get_view_frame_from_road_frame(0, 0, 0, model_height))

model_frame_from_bigmodel_frame = np.dot(model_intrinsics, np.linalg.inv(bigmodel_intrinsics))

# 'camera from model camera'
def get_model_height_transform(camera_frame_from_road_frame, height):
  A = camera_frame_from_road_frame
  height_diff = height - model_height
  x = height_diff * A[0][2]
  y = height_diff * A[1][2]
  z = height_diff * A[2][2]
  eg = A[1][1] * A[2][0]
  dh = A[1][0] * A[2][1]
  ah = A[0][0] * A[2][1]
  bg = A[0][1] * A[2][0]
  bd = A[0][1] * A[1][0]
  ae = A[0][0] * A[1][1]
  det = (
    ae * (z + A[2][3]) +
    bg * (y + A[1][3]) +
    dh * (x + A[0][3]) -
    eg * (x + A[0][3]) -
    bd * (z + A[2][3]) -
    ah * (y + A[1][3])
  )
  eg_dh__det = (eg - dh) / det
  ah_bg__det = (ah - bg) / det
  bd_ae__det = (bd - ae) / det
  high_camera_from_low_camera = np.array([
    [1 + x * eg_dh__det,     x * ah_bg__det,     x * bd_ae__det],
    [    y * eg_dh__det, 1 + y * ah_bg__det,     y * bd_ae__det],
    [    z * eg_dh__det,     z * ah_bg__det, 1 + z * bd_ae__det],
  ])
  return high_camera_from_low_camera


# camera_frame_from_model_frame aka 'warp matrix'
# was: calibration.h/CalibrationTransform
def get_camera_frame_from_model_frame(camera_frame_from_road_frame, height=model_height):
  vp = vp_from_ke(camera_frame_from_road_frame)

  model_camera_from_model_frame = np.array([
    [model_zoom,         0., vp[0] - MODEL_CX * model_zoom],
    [        0., model_zoom, vp[1] - MODEL_CY * model_zoom],
    [        0.,         0.,                            1.],
  ])

  # This function is super slow, so skip it if height is very close to canonical
  if abs(height - model_height) > 0.001: #
    camera_from_model_camera = get_model_height_transform(camera_frame_from_road_frame, height)
  else:
    camera_from_model_camera = np.eye(3)

  return np.dot(camera_from_model_camera, model_camera_from_model_frame)


def get_camera_frame_from_bigmodel_frame(camera_frame_from_road_frame):
  camera_frame_from_ground = camera_frame_from_road_frame[:, (0, 1, 3)]
  bigmodel_frame_from_ground = bigmodel_frame_from_road_frame[:, (0, 1, 3)]

  ground_from_bigmodel_frame = np.linalg.inv(bigmodel_frame_from_ground)
  camera_frame_from_bigmodel_frame = np.dot(camera_frame_from_ground, ground_from_bigmodel_frame)

  return camera_frame_from_bigmodel_frame


def get_model_frame(snu_full, camera_frame_from_model_frame, size):
  idxs = camera_frame_from_model_frame.dot(np.column_stack([np.tile(np.arange(size[0]), size[1]),
                                                            np.tile(np.arange(size[1]), (size[0],1)).T.flatten(),
                                                            np.ones(size[0] * size[1])]).T).T.astype(int)
  calib_flat = snu_full[idxs[:,1], idxs[:,0]]
  if len(snu_full.shape) == 3:
    calib = calib_flat.reshape((size[1], size[0], 3))
  elif len(snu_full.shape) == 2:
    calib = calib_flat.reshape((size[1], size[0]))
  else:
    raise ValueError("shape of input img is weird")
  return calib
