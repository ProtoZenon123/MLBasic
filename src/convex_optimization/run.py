"""Solve a small geometric program with CVXOPT."""

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file


def main() -> None:
    configure_logger(result_file("convex_optimization"))
    logger.info("Convex optimization: started")
    data = pd.read_csv(data_file("convex_optimization", "geometric_program.csv"))
    first, second = data["coefficient"].to_numpy()
    budget = float(data["budget"].iloc[0])
    # In log coordinates z=(log x, log y): minimize a/x+b/y subject to x+y <= budget.
    exponents = matrix(np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]]))
    log_coefficients = matrix(np.log([first, second, 1 / budget, 1 / budget]))
    solvers.options["show_progress"] = False
    solution = solvers.gp([2, 2], exponents, log_coefficients)
    if solution["status"] != "optimal":
        raise RuntimeError(f"CVXOPT status: {solution['status']}")
    x_value, y_value = np.exp(np.array(solution["x"]).ravel())
    objective = first / x_value + second / y_value
    logger.log("RESULT", "Geometric programming (CVXOPT)")
    logger.log("RESULT", "a={:.8f}, b={:.8f}, budget={:.8f}", first, second, budget)
    logger.log("RESULT", "x={:.8f}, y={:.8f}, objective={:.8f}, budget residual={:.8f}",
               x_value, y_value, objective, max(0.0, x_value + y_value - budget))
    logger.info("Convex optimization: completed")


if __name__ == "__main__":
    main()
