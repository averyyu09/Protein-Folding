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



#final protein fold
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

#best protein fold
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
    plt.title("Energy Landscape Exploration")
    plt.grid(True)
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