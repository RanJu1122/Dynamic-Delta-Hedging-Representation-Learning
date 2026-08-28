"""Repository-level API and launcher for the SVI local-volatility task.

The four fixed-signature functions required by the assignment are re-exported
here, while their implementation lives in the installable ``svi_localvol``
package.
"""

from svi_localvol.solution import (ImpliedVol, build_surface, cli, gen_schedule,
                                   localvol, main, param_convert)

__all__ = [
    "param_convert",
    "gen_schedule",
    "ImpliedVol",
    "localvol",
    "build_surface",
    "main",
]


if __name__ == "__main__":
    cli()
