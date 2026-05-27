"""ML Model endpoints - Deprecated.

All endpoints have been migrated to /api/universal/compare
"""

import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ml", tags=["machine learning"])

# Legacy endpoints removed - use /api/universal/compare instead
