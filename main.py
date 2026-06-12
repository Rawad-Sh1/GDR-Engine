from __future__ import annotations

from contextlib import asynccontextmanager
from enum import Enum
from fractions import Fraction
from typing import Final, Literal, Optional

import logging

from fastapi import APIRouter, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

# ============================================================
# Constants
# ============================================================

SERVICE_NAME: Final[str] = "Golden Digital Root API"
API_VERSION: Final[str] = "2.0.0"
THEORY_VERSION: Final[str] = "1.0.0"

MAX_ABS: Final[int] = 10**9
MAX_BATCH_SIZE: Final[int] = 50

GoldenNumber = Literal[3, 6, 9]

INVERTED_DR: Final[dict[int, int]] = {3: 6, 6: 3, 9: 9}

# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("gdr")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# ============================================================
# App Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("%s started | API=%s | Theory=%s", SERVICE_NAME, API_VERSION, THEORY_VERSION)
    yield
    logger.info("%s shutting down", SERVICE_NAME)

app = FastAPI(
    lifespan=lifespan,
    title=SERVICE_NAME,
    description="""
Evaluates and verifies the Golden Digital Root (GDR) equation:

Ng1 × (1 + Ng2 × n)

Golden numbers (Ng1, Ng2): 3, 6, or 9.

Definition (Fractional Digital Root)

Let x be a terminating decimal.

Let k be the smallest non-negative integer such that:

x × 10^k

is an integer.

Then:

GDR(x) = DR(x × 10^k)

Examples:

7.5   → 75   → DR = 3
0.375 → 375  → DR = 6
""".strip(),
    version=API_VERSION,
)

router = APIRouter(prefix="/api/v1", tags=["GDR Engine"])

# ============================================================
# Error Codes
# ============================================================

class ErrorCode(str, Enum):
    VALIDATION_ERROR        = "VALIDATION_ERROR"
    NON_TERMINATING_DECIMAL = "NON_TERMINATING_DECIMAL"
    INTERNAL_SERVER_ERROR   = "INTERNAL_SERVER_ERROR"

# ============================================================
# Models
# ============================================================

class MatrixRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ng1: GoldenNumber = Field(
        ...,
        description="Primary golden number. Must be 3, 6, or 9.",
        json_schema_extra={"example": 3},
    )
    ng2: GoldenNumber = Field(
        ...,
        description="Secondary golden number. Must be 3, 6, or 9.",
        json_schema_extra={"example": 6},
    )
    numerator: int = Field(
        ...,
        ge=-MAX_ABS,
        le=MAX_ABS,
        description=f"Numerator of n, bounded to ±{MAX_ABS:,}.",
        json_schema_extra={"example": 1},
    )
    denominator: int = Field(
        1,
        ge=1,
        le=MAX_ABS,
        description=f"Positive denominator of n, bounded to 1…{MAX_ABS:,}.",
        json_schema_extra={"example": 4},
    )


class BatchRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluations: tuple[MatrixRequest, ...] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"Up to {MAX_BATCH_SIZE} independent GDR evaluations.",
    )


class MatrixState(str, Enum):
    INVALID_FRACTION_INFINITE_DECIMALS = "INVALID_FRACTION_INFINITE_DECIMALS"
    SPECIAL_CASE_ZERO_STATE            = "SPECIAL_CASE_ZERO_STATE"
    VALID_ZERO_STATE                   = "VALID_ZERO_STATE"
    VALID_INTEGER_POSITIVE             = "VALID_INTEGER_POSITIVE"
    VALID_INTEGER_NEGATIVE             = "VALID_INTEGER_NEGATIVE"
    VALID_FRACTION_POSITIVE            = "VALID_FRACTION_POSITIVE"
    VALID_FRACTION_NEGATIVE            = "VALID_FRACTION_NEGATIVE"


class MatrixResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_state:                 MatrixState
    n_value:                 Optional[str]       = None
    zero_crossing_threshold: Optional[str]       = None
    equation:                Optional[str]       = None
    mathematical_result:     Optional[str]       = None
    calculated_digital_root: Optional[int]       = None
    expected_digital_root:   Optional[int]       = None
    logic_verified:          Optional[bool]      = None
    error_code:              Optional[ErrorCode] = None
    message:                 Optional[str]       = None
    note:                    Optional[str]       = None
    api_version:             str                 = API_VERSION
    theory_version:          str                 = THEORY_VERSION


class BatchResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    results:        tuple[MatrixResponse, ...]
    total:          int
    api_version:    str = API_VERSION
    theory_version: str = THEORY_VERSION

# ============================================================
# Exception Handlers
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {"loc": err.get("loc"), "msg": err.get("msg"), "type": err.get("type")}
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_code": ErrorCode.VALIDATION_ERROR.value,
            "message": "Request validation failed.",
            "details": details,
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": ErrorCode.INTERNAL_SERVER_ERROR.value,
            "message": "Unexpected server error.",
        },
    )

# ============================================================
# Utility Functions
# ============================================================

def digital_root(value: int) -> int:
    """O(1) digital root via modulo 9 arithmetic."""
    value = abs(value)
    return 0 if value == 0 else 1 + ((value - 1) % 9)


def analyze_denominator(denominator: int) -> tuple[bool, int, Optional[int]]:
    """
    Returns:
        is_terminating  -> True if denominator has no prime factors other than 2 or 5.
        scale_power     -> Smallest k such that multiplying by 10^k clears the decimal part.
        offending_prime -> Smallest non-2/5 prime factor, or None if terminating.
    """
    d, p, q = denominator, 0, 0

    while d % 2 == 0:
        d //= 2
        p += 1

    while d % 5 == 0:
        d //= 5
        q += 1

    if d == 1:
        return True, max(p, q), None

    i = 3
    while i * i <= d:
        if d % i == 0:
            return False, max(p, q), i
        i += 2

    return False, max(p, q), d


def golden_digital_root(value: Fraction) -> int:
    """
    Golden Digital Root of a terminating Fraction.

    Example:
        15/2 = 7.5
        7.5 -> 75
        DR(75) = 3
    """
    is_terminating, scale, _ = analyze_denominator(value.denominator)

    if not is_terminating:
        raise ValueError(f"{value} is not a terminating decimal.")

    scaled_integer = abs(value.numerator) * (10**scale) // value.denominator
    return digital_root(scaled_integer)


def fmt_equation(ng1: int, ng2: int, n: Fraction, result: Fraction) -> str:
    """Readable equation string."""
    return f"{ng1} × (1 + {ng2} × {n}) = {result}"


def classify_state_and_expected_dr(
    ng1: int,
    n: Fraction,
    threshold: Fraction,
) -> tuple[MatrixState, int, str]:
    """Returns the state, expected digital root, and explanatory note."""
    is_integer = n.denominator == 1

    if n == 0:
        return (
            MatrixState.VALID_ZERO_STATE,
            ng1,
            "n is exactly zero. Digital root maps directly to ng1.",
        )

    if n > 0:
        state = MatrixState.VALID_INTEGER_POSITIVE if is_integer else MatrixState.VALID_FRACTION_POSITIVE
        return state, ng1, "n is positive. Digital root maps directly to ng1."

    state = MatrixState.VALID_INTEGER_NEGATIVE if is_integer else MatrixState.VALID_FRACTION_NEGATIVE

    if n > threshold:
        return (
            state,
            ng1,
            "n is negative but above the zero-crossing threshold. Result remains positive; digital root maps to ng1.",
        )

    return (
        state,
        INVERTED_DR[ng1],
        "n is negative and below the zero-crossing threshold. Result is negative; digital root follows INVERTED_DR.",
    )

# ============================================================
# Core Engine
# ============================================================

def evaluate_gdr(ng1: int, ng2: int, numerator: int, denominator: int) -> MatrixResponse:
    """Pure business logic for the Golden Digital Root theory."""
    n = Fraction(numerator, denominator)
    threshold = Fraction(-1, ng2)

    logger.debug(
        "Evaluating GDR | ng1=%s ng2=%s numerator=%s denominator=%s n=%s",
        ng1, ng2, numerator, denominator, n,
    )

    if n == threshold:
        return MatrixResponse(
            n_state=MatrixState.SPECIAL_CASE_ZERO_STATE,
            n_value=str(n),
            zero_crossing_threshold=str(threshold),
            equation=fmt_equation(ng1, ng2, n, Fraction(0)),
            mathematical_result="0",
            calculated_digital_root=0,
            expected_digital_root=0,
            logic_verified=True,
            message=f"n equals the zero-crossing threshold (-1/{ng2}). Result is 0.",
        )

    is_terminating, _, offending_prime = analyze_denominator(n.denominator)

    if not is_terminating:
        logger.info("Non-terminating input rejected | n=%s offending_prime=%s", n, offending_prime)
        return MatrixResponse(
            n_state=MatrixState.INVALID_FRACTION_INFINITE_DECIMALS,
            n_value=str(n),
            zero_crossing_threshold=str(threshold),
            error_code=ErrorCode.NON_TERMINATING_DECIMAL,
            message=(
                f"Denominator contains prime factor {offending_prime} "
                f"(not 2 or 5), producing a non-terminating decimal."
            ),
        )

    result = Fraction(ng1) * (1 + Fraction(ng2) * n)
    actual_dr = golden_digital_root(result)
    state, expected_dr, note = classify_state_and_expected_dr(ng1, n, threshold)

    verified = actual_dr == expected_dr

    logger.info(
        "GDR evaluation complete | n=%s result=%s actual_dr=%s expected_dr=%s verified=%s",
        n, result, actual_dr, expected_dr, verified,
    )

    return MatrixResponse(
        n_state=state,
        n_value=str(n),
        zero_crossing_threshold=str(threshold),
        equation=fmt_equation(ng1, ng2, n, result),
        mathematical_result=str(result),
        calculated_digital_root=actual_dr,
        expected_digital_root=expected_dr,
        logic_verified=verified,
        note=note,
    )

# ============================================================
# Endpoints
# ============================================================

@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
def health_check():
    return {
        "status": "operational",
        "api_version": API_VERSION,
        "theory_version": THEORY_VERSION,
    }


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": SERVICE_NAME,
        "api_version": API_VERSION,
        "theory_version": THEORY_VERSION,
    }


@router.post(
    "/matrix/evaluate",
    response_model=MatrixResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    summary="Evaluate a single GDR expression",
)
def evaluate_matrix(request: MatrixRequest) -> MatrixResponse:
    return evaluate_gdr(
        ng1=request.ng1,
        ng2=request.ng2,
        numerator=request.numerator,
        denominator=request.denominator,
    )


@router.post(
    "/matrix/batch",
    response_model=BatchResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    summary=f"Evaluate up to {MAX_BATCH_SIZE} GDR expressions in one request",
)
def evaluate_matrix_batch(request: BatchRequest) -> BatchResponse:
    results = tuple(
        evaluate_gdr(
            ng1=item.ng1,
            ng2=item.ng2,
            numerator=item.numerator,
            denominator=item.denominator,
        )
        for item in request.evaluations
    )
    return BatchResponse(results=results, total=len(results))


app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)