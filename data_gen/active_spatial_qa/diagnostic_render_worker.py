"""Compatibility entry for the finite, hash-validated rendering worker."""
import asyncio
from .position_render_run import main

if __name__ == '__main__':
    asyncio.run(main())
