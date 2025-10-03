import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from multiprocessing import Pool, cpu_count
import time
from scipy.ndimage import laplace

plt.style.use('dark_background')

# --- Global function for multiprocessing ---
# This function is now at the top level to ensure it can be pickled by multiprocessing.Pool
def _update_field_chunk_global(args):
    """
    Calculates the next state for a chunk of the field.
    This is a global function to be compatible with multiprocessing.Pool.
    All necessary simulation parameters are passed as arguments.
    """
    field_chunk, memory_chunk, c_eff_chunk, y_start, y_end, step, \
    alpha, beta, kappa, psi_0, field_clip_value, memory_threshold = args
    
    # Pad the chunk to calculate Laplacian correctly at edges
    padded_field = np.pad(field_chunk, 1, mode='wrap') 
    
    # Calculate Laplacian for the chunk
    laplacian_val = laplace(padded_field)[1:-1, 1:-1]
    
    # Unraveling pressure (U)
    unraveling_pressure = kappa * np.abs(field_chunk - psi_0)
    
    # Field update rule
    laplacian_contrib = alpha * c_eff_chunk * laplacian_val
    new_field_chunk = field_chunk + laplacian_contrib - beta * unraveling_pressure
    
    # Clip field values to maintain numerical stability
    new_field_chunk = np.clip(new_field_chunk, -field_clip_value, field_clip_value)

    # Update memory field for this chunk
    memory_increase = np.abs(new_field_chunk - psi_0) * 0.1 
    new_memory_chunk = memory_chunk + memory_increase
    new_memory_chunk = np.clip(new_memory_chunk, 0, memory_threshold * 2) 

    return new_field_chunk, new_memory_chunk


class MassEmergence:
    """
    Phase 4: Watch mass condense from balanced field dynamics
    """
    def __init__(self, width=256, height=128):
        self.width = width
        self.height = height
        
        # Field parameters
        self.alpha = 0.15  # Coherence
        self.beta = 0.08   # Unraveling
        self.kappa = 1.2   # Sensitivity
        self.psi_0 = 0.5   # Equilibrium
        
        # Clipping value for numerical stability
        self.field_clip_value = 100.0 
        
        # Mass emergence parameters
        self.memory_threshold = 0.8  # How much memory before mass forms
        self.recursion_depth = 5     # Compression loops needed for mass (conceptual, not fully implemented for simplicity)
        self.mass_coupling = 0.3     # How strongly mass resists change (used in mass update)
        self.mass_creation_strength = 10.0 # How much mass a particle starts with
        self.min_mass_for_persistence = 0.1 # Minimum mass to be considered stable and kept

        # Initialize fields
        self.field = np.full((height, width), self.psi_0, dtype=np.float64)
        self.initial_field = self.field.copy()
        
        # Memory field: keeps track of persistent coherence
        self.memory_field = np.zeros((height, width), dtype=np.float64)
        
        # Particles: list of (x, y, mass, birth_time)
        self.mass_particles = [] 
        
        # Track metrics
        self.total_mass_history = []
        self.field_std_history = [] # Standard deviation of the field
        self.compression_history = [] # For the whole field (sum of absolute deviations from psi_0)
        self.num_particles_history = []
        
        # Time zones (as in Phase 3)
        self.c_eff = self._create_time_zones()
        
        self.pool = Pool(cpu_count()) # For multiprocessing

    def _create_time_zones(self):
        """
        Creates three vertical time zones with different c_eff values.
        Matches Phase 3 logic: delayed zones allow persistence.
        """
        c_eff_field = np.zeros((self.height, self.width), dtype=np.float64)
        
        # Instant Zone (Left - 1/4 of width)
        c_eff_field[:, :self.width // 4] = 100.0  
        
        # Transition Zone (Middle - 1/2 of width)
        for i in range(self.width // 4, self.width * 3 // 4):
            progress = (i - self.width // 4) / (self.width // 2)
            c_eff_field[:, i] = 100.0 - (95.0 * progress) # Transition from 100 to 5
        
        # Delayed Zone (Right - 1/4 of width)
        c_eff_field[:, self.width * 3 // 4:] = 5.0  
        
        return c_eff_field

    def run_simulation(self, steps):
        """
        Runs the simulation for a given number of steps.
        """
        num_cores = cpu_count()
        chunk_size = self.height // num_cores
        
        print(f"Detected {num_cores} CPU cores - engaging all for maximum performance!")
        print(f"Field: {self.width}x{self.height}")
        print(f"Mass emergence threshold: {self.memory_threshold:.2f} memory accumulation")
        print(f"Recursion depth for mass (conceptual): {self.recursion_depth} cycles")
        print(f"Initial field uniformity (std dev): {np.std(self.field):.6f}")

        # Initial compression event for Phase 4 (as in original code)
        initial_strength = 2.0 
        self.field[self.height//2, self.width//8] += initial_strength # Instant zone
        self.field[self.height//2, self.width//2] += initial_strength # Transition zone
        self.field[self.height//2, self.width*7//8] += initial_strength # Delayed zone
        
        # Apply clipping immediately after initial injection
        self.field = np.clip(self.field, -self.field_clip_value, self.field_clip_value)

        start_time_total = time.time()

        for step in range(steps):
            step_start_time = time.time()
            
            # Prepare chunks for multiprocessing
            chunks_args = []
            for i in range(num_cores):
                y_start = i * chunk_size
                y_end = (i + 1) * chunk_size if i < num_cores - 1 else self.height
                chunks_args.append((self.field[y_start:y_end], 
                                     self.memory_field[y_start:y_end], 
                                     self.c_eff[y_start:y_end], 
                                     y_start, y_end, step,
                                     # Pass all necessary parameters for the global function:
                                     self.alpha, self.beta, self.kappa, self.psi_0, 
                                     self.field_clip_value, self.memory_threshold))

            # Apply updates in parallel using the global function
            results = self.pool.map(_update_field_chunk_global, chunks_args)

            # Reconstruct field and memory from chunks
            for i, (new_field_chunk, new_memory_chunk) in enumerate(results):
                y_start = i * chunk_size
                y_end = (i + 1) * chunk_size if i < num_cores - 1 else self.height
                self.field[y_start:y_end] = new_field_chunk
                self.memory_field[y_start:y_end] = new_memory_chunk

            # --- Mass Emergence Logic ---
            candidate_coords = np.argwhere(self.memory_field > self.memory_threshold)
            
            newly_formed_particles = []
            existing_particle_coords = [(int(p[0]), int(p[1])) for p in self.mass_particles if p[2] >= self.min_mass_for_persistence]

            for r, c in candidate_coords:
                # Check proximity to existing stable particles
                too_close = False
                for px, py in existing_particle_coords:
                    dist_sq = (px - c)**2 + (py - r)**2
                    if dist_sq < (5.0)**2: 
                        too_close = True
                        break
                
                if not too_close:
                    mass_val = self.mass_creation_strength 
                    newly_formed_particles.append((c, r, mass_val, step)) 
                    
                    # Consume memory at the particle location
                    self.memory_field[r, c] *= 0.5 
                    
                    print(f"  PARTICLE BORN at ({c}, {r}) with mass {mass_val:.3f}!")

            self.mass_particles.extend(newly_formed_particles)

            # --- Mass Interaction: Update particle mass based on local field coherence ---
            updated_particles = []
            for i, particle in enumerate(self.mass_particles):
                px, py, pm, pt = particle
                px_int, py_int = int(round(px)), int(round(py)) 

                # Check bounds for safety
                if 0 <= py_int < self.height and 0 <= px_int < self.width:
                    local_coherence = self.field[py_int, px_int]
                    
                    # Mass grows if local coherence is high, decays otherwise
                    if local_coherence > self.psi_0 + 0.1: 
                        pm += 0.01 * (local_coherence - self.psi_0) * self.mass_coupling
                    else: 
                        pm -= 0.005 * self.mass_coupling
                    
                    pm = np.clip(pm, self.min_mass_for_persistence, 50.0) 
                    
                    if pm >= self.min_mass_for_persistence: 
                        updated_particles.append((px, py, pm, pt))
                else:
                    pass 
            self.mass_particles = updated_particles

            # Store metrics
            current_total_mass = sum(p[2] for p in self.mass_particles)
            self.total_mass_history.append(current_total_mass)
            self.field_std_history.append(np.std(self.field))
            self.compression_history.append(np.sum(np.abs(self.field - self.psi_0))) 
            self.num_particles_history.append(len(self.mass_particles))

            step_end_time = time.time()
            steps_per_sec = 1 / (step_end_time - step_start_time) if (step_end_time - step_start_time) > 0 else float('inf')
            print(f"Step {step+1}/{steps} - {steps_per_sec:.1f} steps/sec - Total mass: {current_total_mass:.2f}, Particles: {self.num_particles_history[-1]}")
            
        end_time_total = time.time()
        print(f"\nSimulation finished in {end_time_total - start_time_total:.2f} seconds.")
        self.pool.close()
        self.pool.join()

    def visualize_results(self):
        """
        Visualizes the simulation results including field state, memory, particles, and historical metrics.
        Improved layout and clarity.
        """
        fig = plt.figure(figsize=(18, 12)) 
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7]) 

        # Subplot 1: Final Field State
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(self.field, cmap='RdBu', origin='lower', 
                         vmin=self.psi_0 - 2, vmax=self.psi_0 + 2, 
                         extent=[0, self.width, 0, self.height])
        ax1.set_title('Final Field State (Ψ)', fontsize=14)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        fig.colorbar(im1, ax=ax1, label='Field Value (Ψ)')
        
        # Add time zone markers
        ax1.axvline(self.width // 4, color='white', linestyle='--', alpha=0.7, label='Zone Boundary')
        ax1.axvline(self.width * 3 // 4, color='white', linestyle='--', alpha=0.7)
        ax1.text(self.width//8, self.height*0.95, 'Instant', color='white', ha='center', va='top', fontsize=10)
        ax1.text(self.width//2, self.height*0.95, 'Transition', color='white', ha='center', va='top', fontsize=10)
        ax1.text(self.width*7//8, self.height*0.95, 'Delayed', color='white', ha='center', va='top', fontsize=10)


        # Subplot 2: Memory Field
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(self.memory_field, cmap='Greys', origin='lower', 
                         vmin=0, vmax=self.memory_threshold * 1.5, 
                         extent=[0, self.width, 0, self.height])
        ax2.set_title('Memory Field (Accumulated Coherence)', fontsize=14)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        fig.colorbar(im2, ax=ax2, label='Memory Value')
        
        # Add time zone markers to memory field as well
        ax2.axvline(self.width // 4, color='white', linestyle='--', alpha=0.7)
        ax2.axvline(self.width * 3 // 4, color='white', linestyle='--', alpha=0.7)

        # Subplot 3: Particle Locations and Mass
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(self.field, cmap='RdBu', origin='lower', alpha=0.2, 
                   vmin=self.psi_0 - 0.5, vmax=self.psi_0 + 0.5,
                   extent=[0, self.width, 0, self.height])
        
        if self.mass_particles:
            birth_times = [p[3] for p in self.mass_particles]
            if birth_times: 
                norm = plt.Normalize(min(birth_times), max(birth_times) + 1)
                cmap = plt.cm.viridis

                for px, py, pm, pt in self.mass_particles:
                    radius = 0.5 + (pm / self.mass_creation_strength) * 2.0 
                    radius = np.clip(radius, 0.5, 5.0) 
                    circle_color = cmap(norm(pt))
                    circle = Circle((px, py), radius=radius, color=circle_color, alpha=0.8, ec='white', lw=0.5)
                    ax3.add_patch(circle)
                
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([]) 
                cbar = fig.colorbar(sm, ax=ax3, label='Particle Birth Time (Simulation Step)')
                cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True)) 
            else: 
                 for px, py, pm, pt in self.mass_particles:
                    radius = 0.5 + (pm / self.mass_creation_strength) * 2.0
                    radius = np.clip(radius, 0.5, 5.0) 
                    circle = Circle((px, py), radius=radius, color='cyan', alpha=0.8, ec='white', lw=0.5)
                    ax3.add_patch(circle)

        ax3.set_title('Emergent Mass Particles', fontsize=14)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_xlim(0, self.width)
        ax3.set_ylim(0, self.height)
        ax3.set_aspect('equal', adjustable='box') 

        # Add time zone markers
        ax3.axvline(self.width // 4, color='white', linestyle='--', alpha=0.7)
        ax3.axvline(self.width * 3 // 4, color='white', linestyle='--', alpha=0.7)
        ax3.text(self.width//8, self.height*0.95, 'Fast', color='white', ha='center', va='top', fontsize=10)
        ax3.text(self.width*7//8, self.height*0.95, 'Slow', color='white', ha='center', va='top', fontsize=10)

        # Subplot 4: Total Mass History
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(self.total_mass_history, color='lime', linewidth=2)
        ax4.set_title('Total Emergent Mass Over Time', fontsize=14)
        ax4.set_xlabel('Simulation Step')
        ax4.set_ylabel('Total Mass')
        ax4.grid(True, linestyle='--', alpha=0.6)

        # Subplot 5: Field Standard Deviation History (Compression)
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(self.field_std_history, color='orange', linewidth=2)
        ax5.set_title('Field Standard Deviation (Compression)', fontsize=14)
        ax5.set_xlabel('Simulation Step')
        ax5.set_ylabel('Std Dev (σ)')
        ax5.grid(True, linestyle='--', alpha=0.6)

        # Subplot 6: Number of Particles History
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(self.num_particles_history, color='cyan', linewidth=2)
        ax6.set_title('Number of Emergent Particles Over Time', fontsize=14)
        ax6.set_xlabel('Simulation Step')
        ax6.set_ylabel('Particle Count')
        ax6.grid(True, linestyle='--', alpha=0.6)

        plt.suptitle('FIELD-AWARE COSMOLOGY: PHASE 4 - MASS EMERGENCE\\n' + 
                     '\"Mass condenses from stable coherence: ethical dynamics form matter\"',
                     fontsize=18, y=1.02) 

        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
        return fig

def main():
    """
    Run Phase 4: Mass Emergence simulation
    """
    print("=== FIELD-AWARE COSMOLOGY: PHASE 4 - MASS EMERGENCE ===")
    print("Demonstrating that mass condenses from ethical field dynamics")
    print("Only balanced regions can create persistent matter!")
    
    # Create and run
    sim = MassEmergence(width=256, height=128)
    sim.run_simulation(steps=300)
    
    # Analysis
    print("\n=== MASS ANALYSIS ===")
    total_mass = sim.total_mass_history[-1] if sim.total_mass_history else 0
    stable_particles = len(sim.mass_particles)
    print(f"Total emergent mass: {total_mass:.2f}")
    print(f"Stable particles formed: {stable_particles}")
    print(f"\nParticle distribution (threshold mass > {sim.min_mass_for_persistence:.2f}):")
    
    persisting_particles = [p for p in sim.mass_particles if p[2] >= sim.min_mass_for_persistence]

    instant_zone_end_x = sim.width // 4
    slow_particles_count = sum(1 for x, y, m, t in persisting_particles 
                               if x >= instant_zone_end_x) 
    fast_particles_count = sum(1 for x, y, m, t in persisting_particles 
                               if x < instant_zone_end_x)
    
    print(f"  In 'Instant' (Fast) time zone: {fast_particles_count}")
    print(f"  In 'Transition/Delayed' (Slow) time zones: {slow_particles_count}")
    
    # Conclusion based on observations
    if slow_particles_count > fast_particles_count and slow_particles_count > 0:
        print(f"\nPROVEN: Mass requires time, memory, and ethical balance (more in slow/delayed zones)!")
    elif stable_particles > 0:
        print(f"\nMass formed! Distribution needs further analysis to confirm zone preference.")
    else:
        print(f"\nNo stable mass particles formed in this run. Adjust parameters for emergence.")

    # Visualize
    fig = sim.visualize_results()
    
    # Save
    output_file = 'fac_phase4_mass_emergence_cleaned.png' 
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"\nSaved visualization to: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    main()