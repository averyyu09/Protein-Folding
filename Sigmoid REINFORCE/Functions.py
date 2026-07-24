import numpy as np
import matplotlib.pyplot as plt

def laydowncoordinates(d):
    coordinates = [(0, 0)]

    for move in d:
        x, y = coordinates[-1]

        if move == 0:
            coordinates.append((x - 1, y))
        elif move == 1:
            coordinates.append((x + 1, y))
        elif move == 2:
            coordinates.append((x, y + 1))
        elif move == 3:
            coordinates.append((x, y - 1))

    return coordinates



def compute_A(Gamma):
    N = len(Gamma)
    A = np.zeros(N, dtype=int)

    for i in range(N):
        if i == 0:
            A[i] = Gamma[i + 1]
        elif i == N - 1:
            A[i] = Gamma[i - 1]
        else:
            A[i] = Gamma[i - 1] + Gamma[i + 1]

    return A



def M(x, y, coordinates, Gamma, H):

    for i in range(len(Gamma)):
        if coordinates[i] == (x, y) and Gamma[i] == H:
            return 1

    return 0



def Mhat(i, coordinates, Gamma, H):

    x, y = coordinates[i]

    return (
        M(x - 1, y, coordinates, Gamma, H)
        + M(x + 1, y, coordinates, Gamma, H)
        + M(x, y - 1, coordinates, Gamma, H)
        + M(x, y + 1, coordinates, Gamma, H)
    )



def energy(coordinates, Gamma, A, H):

    E = 0

    for i in range(len(Gamma)):
        E += Gamma[i] * (A[i] - Mhat(i, coordinates, Gamma, H))

    return E / 2



def valid_fold(coordinates):
    return len(set(coordinates)) == len(coordinates)

def propose_move(d):

    d_new = d.copy()

    i = np.random.randint(len(d))

    current = d_new[i]

    choices = [0, 1, 2, 3]
    choices.remove(current)

    d_new[i] = np.random.choice(choices)

    return d_new






def plot_results(energies, best_energies, coordinates,best_coordinates,Gamma,H,best_E,):
    plt.figure(figsize=(8, 5))
    plt.plot(best_energies)
    plt.xlabel("Iteration")
    plt.ylabel("Best Energy")
    plt.title("Best Energy vs Iteration")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(energies)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Simulated Annealing")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(energies, bins=20)
    plt.xlabel("Energy")
    plt.ylabel("Frequency")
    plt.title("Histogram of Sampled Energies")
    plt.show()

#final fold
    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]

    plt.figure(figsize=(6, 6))

    plt.plot(x, y, "-k", linewidth=2)

    for i in range(len(Gamma)):

        if Gamma[i] == H:
            plt.scatter(x[i], y[i], s=150, color="red")
        else:
            plt.scatter(x[i], y[i], s=150, color="blue")

        plt.text(x[i] + 0.1, y[i] + 0.1, str(i), fontsize=10)

    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Final HP Protein Fold")
    plt.show()

#best fold
    x = [coord[0] for coord in best_coordinates]
    y = [coord[1] for coord in best_coordinates]

    colors = ["red" if g == H else "blue" for g in Gamma]

    plt.figure(figsize=(6, 6))

    plt.plot(x, y, "-k", linewidth=2)

    plt.scatter(
        x,
        y,
        c=colors,
        s=200,
        edgecolors="black",
    )

    for i, (xi, yi) in enumerate(best_coordinates):
        plt.text(xi + 0.1, yi + 0.1, str(i), fontsize=9)

    plt.axis("equal")
    plt.grid(True)
    plt.title(f"Best Fold Found (Energy = {best_E})")
    plt.show()

def plot_energy_uniquefolds(uniquefold_energies):
    plt.figure(figsize=(8,4))
    plt.plot(uniquefold_energies)
    plt.xlabel("Unique fold discovered")
    plt.ylabel("Energy")
    plt.title("Energy per new fold")
    plt.grid(True)
    plt.show()

def plot_uniquefolds_incontext(which_sweeps,uniquefold_energies):

    fig, ax = plt.subplots(figsize=(8,4))

    # Bottom axis: unique fold number
    ax.plot(uniquefold_energies, '-')
    ax.set_xlabel("Unique fold discovered")
    ax.set_ylabel("Energy")
    ax.set_title("Energy of Newly Discovered Folds")
    ax.grid(True)

    # Top axis: Monte Carlo sweep
    ax2 = ax.twiny()

    ax2.set_xlim(ax.get_xlim())

    # Put a few evenly spaced ticks
    tick_positions = np.linspace(
        0,
        len(which_sweeps)-1,
        6,
        dtype=int
    )

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([which_sweeps[i] for i in tick_positions])
    ax2.set_xlabel("Monte Carlo Sweep")

    plt.show()


def energywithpenalties(Gamma, coordinates):

    N = len(Gamma)

    E_hp = 0
    E_overlap = 0
    E_bond = 0

    # overlap penalty
    for i in range(N):
        for j in range(i+1, N):
            if np.array_equal(coordinates[i], coordinates[j]):
                E_overlap += 1

    # bond penalty
    for i in range(N-1):

        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i+1]

        d = abs(x1-x2) + abs(y1-y2)

        E_bond += (d-1)**2

    # HP energy
    for i in range(N):
        for j in range(i+2, N):

            xi, yi = coordinates[i]
            xj, yj = coordinates[j]

            if abs(xi-xj) + abs(yi-yj) == 1:
                E_hp -= Gamma[i]*Gamma[j]

    return E_hp + 100*E_overlap + 100*E_bond



def generate_all_proposals(d):
    proposals = []
    for i in range(len(d)):
        for direction in range(4):
            if direction == d[i]:
                continue
            d_new = d.copy()
            d_new[i] = direction
            proposals.append(d_new)
    return proposals


def generate_legal_neighbors(d):
    legal = []
    for proposal in generate_all_proposals(d):
        coords = laydowncoordinates(proposal)
        if valid_fold(coords):
            legal.append(proposal)
    return legal


def barker(deltaE, beta):
    return 1 / (1 + np.exp(beta * deltaE))


def cost_per_sample(Gamma, A, H, d):
    coords = laydowncoordinates(d)
    return energy(coords, Gamma, A, H)



#compute, for a trajectory, the end energy, the ending configuration, and the derivative of log P
def RL_trajectory(d0, Gamma, A, H, beta, num_time_steps):
    #initiate current cost, energy, gradient
    d = d0.copy()
    E_current = cost_per_sample(Gamma, A, H, d)
    grad_logP = 0.0
    
    #do this for the number of steps we want to look into the future:
    #each iteration, accept or reject the new proposed fold, then add that probability that it happened to the gradient
    for t in range(num_time_steps):
        #generate all legal next moves
        neighbors = generate_legal_neighbors(d)
        if len(neighbors) == 0:
            break  #dead end 
        # out of the possible legal next moves, choose 1
        j = np.random.randint(len(neighbors))
        d_prime = neighbors[j]
        E_prime = cost_per_sample(Gamma, A, H, d_prime)
        deltaE = E_prime - E_current
        #accept the proposed state or stay at current state?
        A_accept = barker(deltaE, beta)
        u = np.random.uniform(0, 1)
        if u < A_accept:
            grad_logP += -deltaE / (1 + np.exp(-beta * deltaE)) #this is the derivative of 1/1+e^beta*E, making the gradient more negative if we accept the new fold
            d = d_prime
            E_current = E_prime
        else:
            grad_logP += deltaE / (1 + np.exp(beta * deltaE)) #this is the derivative of 1/1+e^beta*E, making the gradient more positive if we reject the new fold

    return d, E_current, grad_logP


def grad_expectedcost(d0, Gamma, beta, num_time_steps=1, num_independent_experiments=10):
    H = 1
    A = compute_A(Gamma)
    costs = np.zeros(num_independent_experiments)
    grad_beta = 0.0
    #for each experiment, beta is updated
    for i in range(num_independent_experiments):
        d_final, E_final, grad_logP = RL_trajectory(d0, Gamma, A, H, beta, num_time_steps)
        costs[i] = E_final
        grad_beta += E_final * grad_logP

    grad_beta /= num_independent_experiments
    return costs, grad_beta


def update_beta(beta, grad_beta, learning_rate):
    return beta - learning_rate * grad_beta

