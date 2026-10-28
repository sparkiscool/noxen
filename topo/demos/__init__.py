"""Demo parts for topo. Each entry returns dict(setup=f(h) -> Part, h=default cell size, gen_cm3=generated volume target).

  holder      the engine's benchmark holder, 28 load cases, 3 supports
  motor_mount generic drone motor mount on a carbon plate (dimensions are assumptions), thrust / torque / crash
"""


def _holder():
    from .holder import setup
    return dict(setup=setup, h=1.0, gen_cm3=6.0, baseline=None)


def _motor():
    from .motor_mount import setup
    return dict(setup=setup, h=0.8, gen_cm3=7.0, baseline=None)


PARTS = {
    "holder": _holder,
    "motor_mount": _motor,
}
