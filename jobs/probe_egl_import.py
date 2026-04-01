import os

print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")
backend = os.environ.get("MUJOCO_GL", "egl")
print(f"MUJOCO_GL={backend}")

if backend == "osmesa":
    from OpenGL import osmesa  # noqa: F401

    print("OSMESA_IMPORT_OK")
else:
    import OpenGL.EGL  # noqa: F401

    print("EGL_IMPORT_OK")

from dm_control import mujoco  # noqa: F401

print("DM_CONTROL_OK")
