"""
name:Aseel Ibrahim Kamel
ID: 0243183
name:REMAS MAJDI MOHAMMAD ABUAWIDA
ID:2240015
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math


class CitadelSimulation:

    def __init__(self, L=200.0, alpha=0.025, delta_x=1.0, tolerance=0.1):
        """
                CLOSE TO GATE YES/NO
        """
        self.L = L  # Total length of the perimeter
        self.alpha = alpha  # How smart the walker is (bias factor)
        self.delta_x = delta_x  # Step size (1 meter)
        self.tolerance = tolerance  # The "Winning Zone" size
        self.start_pos = L / 2  # Start exactly opposite the gate

        # Create the map shape immediately
        self.street_x, self.street_y = self.create_street_contour()

    def create_street_contour(self):
        """to create an irregular closed shape representing the citadel map"""

        theta = list(np.arange(0, 361,10))
        street_x = []
        for t in theta:
            street_x.append((self.L / 2 * np.cos(t * (math.pi / 180))))

        street_y = []
        for t in theta:
            street_y.append((self.L / 2 * np.sin(t * (math.pi / 180))) + random.randint(-10, 10))

        street_y[0] = 0
        street_y[-1] = 0

        return street_x , street_y


    def d_left(self, x):
        """To calculate the distance between the entrance and current position (Clock wise)"""
        return x % self.L

    def d_right(self, x):
        """To calculate the distance between the entrance and current position (Counter-Clockwise)"""
        return (self.L - x) % self.L

    def prob_right(self, x):
        """ To Calculate the probability of moving right"""
        return 1 / (1 + math.exp(-self.alpha * (self.d_left(x) - self.d_right(x))))

    def run_simulation(self,force_direction=None):
        """To simulate the steps taken from the start till reaching the entrance, considering the human errors that might occur"""
        delta_x = 1
        X = self.L / 2

        positions = [X]
        probabilities = [self.prob_right(X)]

        if force_direction == 'right':
            X = X + 10 * delta_x
        elif force_direction == 'left':
            X = X - 10 * delta_x

        while self.d_left(X) > self.tolerance:
            p = self.prob_right(X)
            r = random.random()
            if r <= p:
                X = X + delta_x
            elif r > p:
                X = X - delta_x

            positions.append(X)
            probabilities.append(p)
        return positions,probabilities

    def get_street_position(self, perimeter_pos):
        """To find the rectangular coordinates (x,y) for each linear distance from the entrance"""

        #to calculate the ratio between degrees and delta_x
        degree = (360/10) * self.delta_x / self.L
        return self.street_x[int(degree*perimeter_pos)], self.street_y[int(degree*perimeter_pos)]

    def animate_results(self):
        """
        Runs both simulations and creates the dual plot animation.
        """
        # Run 1: Force Left first
        print("Running first simulation (forced clockwise)...")
        pos1, probs1 = self.run_simulation('left')

        # Run 2: Force Right second
        print("Running second simulation (forced counter-clockwise)...")
        pos2, probs2 = self.run_simulation('right')

        # Combine the data for one long animation
        all_positions = pos1 + pos2
        all_probs = probs1 + probs2
        split_index = len(pos1)
        total_frames = len(all_positions)

        print(f"\nFirst simulation: {len(pos1) - 1} steps")
        print(f"Second simulation: {len(pos2) - 1} steps")
        print(f"Total animation frames: {total_frames}")

        # --- Plotting Setup ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Finding Amman's Citadel Entrance\nStudent: REMAS & ASEEL", fontsize=14)

        # LEFT PLOT: The Map
        ax1.set_title("Movement Around Amman's Citadel")
        ax1.plot(self.street_x, self.street_y, 'gray', lw=3, alpha=0.7, label='Citadel Perimeter')

        # Mark the entrance
        ent_x, ent_y = self.get_street_position(0)
        ax1.plot(ent_x, ent_y, 'go', ms=10, label='Entrance (x=0)')

        # Initialize walker and path
        walker_dot, = ax1.plot([], [], 'ro', ms=8, label='Walker')
        path_line, = ax1.plot([], [], 'r-', lw=2, alpha=0.6, label='Path')

        ax1.legend(loc='upper right')
        ax1.axis('equal')
        ax1.axis('off')

        # RIGHT PLOT: Probability Graph
        ax2.set_title("Probability Evolution (P of going Right)")
        ax2.set_xlabel("Step Number")
        ax2.set_ylabel("P(Go Counter-clockwise)")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlim(0, total_frames + 10)
        ax2.grid(True, alpha=0.3)

        prob_line, = ax2.plot([], [], 'b-', lw=2)
        ax2.axhline(0.5, color='gray', ls='--', alpha=0.5)
        ax2.axvline(split_index, color='brown', ls=':', lw=2, label='Sim 2 Start')

        status_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)

        # Animation Functions
        def init():
            walker_dot.set_data([], [])
            path_line.set_data([], [])
            prob_line.set_data([], [])
            status_text.set_text('')
            return walker_dot, path_line, prob_line, status_text

        def update(frame):
            # Update Walker position
            curr_pos = all_positions[frame]
            wx, wy = self.get_street_position(curr_pos)
            walker_dot.set_data([wx], [wy])

            # Decide which simulation we are drawing
            if frame < split_index:
                segment = all_positions[0: frame + 1]
                sim_name = "Sim 1: Forced Clockwise"
            else:
                segment = all_positions[split_index: frame + 1]
                sim_name = "Sim 2: Forced Counter-CW"

            # Draw the red trail
            tx, ty = [], []
            for p in segment:
                px, py = self.get_street_position(p)
                tx.append(px)
                ty.append(py)
            path_line.set_data(tx, ty)

            # Update the probability graph
            curr_prob_len = min(frame, len(all_probs))
            prob_line.set_data(range(curr_prob_len), all_probs[:curr_prob_len])

            status_text.set_text(f"{sim_name}\nStep: {frame}")

            return walker_dot, path_line, prob_line, status_text

        # Start Animation
        ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                      init_func=init, blit=True, interval=30, repeat=False)
        plt.tight_layout()
        plt.show(block=True)


# Main Execution Block
if __name__ == "__main__":
    citadel = CitadelSimulation()
    citadel.animate_results()
