"""Save the scene as a self-contained .blend that animates on its own.

    python make_blend.py OUT.blend

geom.py and scene.py go into the file as text blocks; a third text block,
rope_driver.py, is registered (runs on load when scripts are allowed) and
poses everything for the current frame from a frame-change handler, so the
timeline can be scrubbed, played and rendered inside Blender.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402
import matplotlib  # noqa: E402

import geom as g  # noqa: E402
import scene  # noqa: E402

DRIVER = '''"""Poses the rope scene for the current frame (Assignment 6, pp-21).

Loads the text blocks geom.py and scene.py as modules and registers a
frame-change handler.  The on-screen captions and distance readouts of
the finished video are added afterwards, outside Blender.
"""
import sys
import types

import bpy


def _load(name):
    mod = types.ModuleType(name)
    mod.__file__ = name + ".py"
    sys.modules[name] = mod
    exec(compile(bpy.data.texts[name + ".py"].as_string(), name + ".py", "exec"), mod.__dict__)
    return mod


geom = _load("geom")
scene = _load("scene")
scene.rebind()


def rope_frame(sc, *args):
    scene.apply(geom.state(sc.frame_current))


for h in list(bpy.app.handlers.frame_change_pre):
    if getattr(h, "__name__", "") == "rope_frame":
        bpy.app.handlers.frame_change_pre.remove(h)
bpy.app.handlers.frame_change_pre.append(rope_frame)
rope_frame(bpy.context.scene)
'''


def main():
    out = os.path.abspath(sys.argv[-1])
    font_dir = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    scene.build(font_dir)
    sc = bpy.context.scene
    sc.frame_start = 0
    sc.frame_end = int(round(g.END * g.FPS)) - 1
    sc.render.fps = g.FPS
    sc.render.use_lock_interface = True
    sc.render.filepath = "//frames/f"
    for f in bpy.data.fonts:
        if f.filepath and f.filepath != "<builtin>":
            f.pack()
    for name in ("geom.py", "scene.py"):
        t = bpy.data.texts.new(name)
        t.from_string(open(os.path.join(HERE, name)).read())
    drv = bpy.data.texts.new("rope_driver.py")
    drv.from_string(DRIVER)
    drv.use_module = True
    sc.frame_current = 0
    scene.apply(g.state(0))
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    print("saved", out)


main()
