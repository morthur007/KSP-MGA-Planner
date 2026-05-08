# Frame contract

## Proven frame mapping

The SPK/PyKEP candidate states are Sun-centered LevelA states.

The Principia impulse particle server expects and returns raw Barycentric absolute states.

Therefore, to inject a SPK/PyKEP state into the impulse server:

    raw_abs = Sun_raw_abs + LevelA_to_raw(spk_rel_sun)

where:

    LevelA_to_raw = +Z,-X,+Y

The opposite transform:

    raw_to_LevelA = -Y,+Z,+X

is used by the Principia exporter when writing LevelA/SPK-style columns.

## Proof

The geometry proof for rank 1, leg 1 showed:

    transform +Z,-X,+Y
    start body mapping error = 0.000009 m
    end body mapping error   = 0.000008 m

while identity and raw_to_LevelA were off by tens of millions of km.

Thus +Z,-X,+Y is the only valid transform for injecting SPK/PyKEP Sun-centered states into the raw Barycentric Principia particle server.
