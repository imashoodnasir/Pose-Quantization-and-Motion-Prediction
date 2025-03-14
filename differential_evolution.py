from scipy.optimize import differential_evolution

def optimize_loss(loss_function, bounds):
    result = differential_evolution(loss_function, bounds)
    return result.x

if __name__ == "__main__":
    bounds = [(0, 1)] * 3  # Example bounds for gamma1, gamma2, gamma3
    optimal_params = optimize_loss(lambda x: x[0] + x[1] + x[2], bounds)
    print("Optimized Params:", optimal_params)
