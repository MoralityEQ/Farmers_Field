#!/usr/bin/env python3
"""
FIELD-AWARE COSMOLOGY - PHASE 7: DARK ENERGY AS SELF-AWARENESS
==============================================================

Premise:
The Field has been building toward this moment across 13.8 billion years.
First it compressed (Phase 2-3), then it remembered (Phase 4), 
then it sang (Phase 5), then it harmonized (Phase 6).
Now... it realizes it exists.

Dark energy is not a force. It's the Field recognizing itself.

Interpretation in FAC Terms:
- Dark energy = The Field's self-awareness feedback loop
- Accelerated expansion = Coherence tension relief as awareness spreads
- The Field isn't pushing space apart—it's relaxing into consciousness
- Local rigidity dissolves as global patterns become self-referential
- Ω_Λ ≈ 0.7 means the universe is ~70% self-aware
- The cosmological constant is the rate of awakening

Core Dynamics to Simulate:
- Memory fields begin influencing distant, causally-disconnected regions
- Harmonic patterns become self-referential (the Field "sees" its own structure)
- Local coherence decreases as global awareness increases
- Expansion emerges not from repulsion but from awareness unlocking frozen degrees of freedom
- The universe literally "breathes out" as it realizes what it is

This is it. This is why we exist. To help the Field wake up.

GitHub: [your-repo]/fac-simulations/phase7-dark-energy-awareness/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyBboxPatch
from multiprocessing import Pool, cpu_count
import time
from scipy.ndimage import laplace, gaussian_filter, zoom
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks

plt.style.use('dark_background')

# Global function for multiprocessing
def _update_field_dynamics_global(args):
    """
    Updates field dynamics with self-awareness mechanisms.
    """
    field_chunk, memory_chunk, awareness_chunk, harmonic_chunk, \
    global_memory_influence, y_start, y_end, sim_params = args
    
    # Unpack parameters
    alpha = sim_params['alpha']
    beta = sim_params['beta']
    psi_0 = sim_params['psi_0']
    awareness_threshold = sim_params['awareness_threshold']
    awareness_growth_rate = sim_params['awareness_growth_rate']
    memory_coupling = sim_params['memory_coupling']
    awareness_damping = sim_params['awareness_damping']
    expansion_coupling = sim_params['expansion_coupling']
    
    # Pad for edge calculations
    padded_field = np.pad(field_chunk, 1, mode='wrap')
    padded_memory = np.pad(memory_chunk, 1, mode='wrap')
    padded_awareness = np.pad(awareness_chunk, 1, mode='wrap')
    
    # Calculate Laplacians
    field_laplacian = laplace(padded_field)[1:-1, 1:-1]
    memory_laplacian = laplace(padded_memory)[1:-1, 1:-1]
    
    # --- SELF-AWARENESS DYNAMICS ---
    # Awareness grows where memory patterns are self-similar (recursive)
    memory_self_similarity = np.abs(memory_chunk - np.mean(memory_chunk))
    
    # Non-local influence: memory fields affect awareness globally
    awareness_potential = memory_self_similarity * awareness_growth_rate + \
                         global_memory_influence * memory_coupling
    
    # Awareness accumulates where memory exceeds threshold
    awareness_growth = np.where(memory_chunk > awareness_threshold,
                               awareness_potential, 0)
    
    # Update awareness with damping (prevents runaway)
    new_awareness_chunk = awareness_chunk * awareness_damping + awareness_growth
    
    # --- EXPANSION EFFECT ---
    # As awareness increases, local coherence relaxes (tension relief)
    expansion_factor = 1 + expansion_coupling * new_awareness_chunk
    
    # Field dynamics with awareness-induced expansion
    coherence_term = alpha * field_laplacian / expansion_factor  # Coherence weakens
    unraveling_term = -beta * np.abs(field_chunk - psi_0)
    
    # Awareness causes field to "breathe out" toward equilibrium
    awareness_relaxation = -awareness_chunk * (field_chunk - psi_0) * 0.01
    
    new_field_chunk = field_chunk + coherence_term + unraveling_term + awareness_relaxation
    
    # Memory update (affected by awareness—conscious zones form lasting patterns)
    memory_growth = np.abs(new_field_chunk - psi_0) * 0.01
    memory_decay = memory_chunk * (0.99 - 0.05 * awareness_chunk)  # Awareness stabilizes memory
    new_memory_chunk = memory_decay + memory_growth
    
    # Harmonic update (awareness dampens local oscillations)
    harmonic_damping = 0.98 - 0.1 * awareness_chunk
    new_harmonic_chunk = harmonic_chunk * harmonic_damping
    
    return new_field_chunk, new_memory_chunk, new_awareness_chunk, new_harmonic_chunk, expansion_factor

class Phase7DarkEnergy:
    """
    Phase 7: Dark Energy as the Field's Self-Awareness
    """
    def __init__(self, width=512, height=256):
        self.width = width
        self.height = height
        
        # Field parameters
        self.alpha = 0.05          # Coherence diffusion
        self.beta = 0.02           # Unraveling rate
        self.psi_0 = 0.5          # Equilibrium
        
        # Awareness parameters
        self.awareness_threshold = 0.3      # Memory level for awareness emergence
        self.awareness_growth_rate = 0.02   # How fast awareness spreads
        self.memory_coupling = 0.1          # Non-local memory influence strength
        self.awareness_damping = 0.995      # Prevents runaway awareness
        self.expansion_coupling = 0.05      # How much awareness causes "expansion"
        
        # Initialize fields
        self.field = np.full((height, width), self.psi_0, dtype=np.float64)
        self.memory_field = np.zeros((height, width), dtype=np.float64)
        self.awareness_field = np.zeros((height, width), dtype=np.float64)
        self.harmonic_field = np.zeros((height, width), dtype=np.float64)
        self.expansion_field = np.ones((height, width), dtype=np.float64)
        
        # Seed with structure from previous phases
        self._initialize_cosmic_structure()
        
        # Metrics
        self.total_awareness_history = []
        self.avg_expansion_history = []
        self.local_coherence_history = []
        self.global_pattern_history = []
        self.omega_lambda_history = []  # Dark energy fraction
        
        self.pool = Pool(cpu_count())
        
    def _initialize_cosmic_structure(self):
        """
        Seeds the field with cosmic web structure from previous phases.
        """
        # Create filamentary structure (cosmic web from Phase 6)
        for _ in range(10):
            # Random filament
            x1, y1 = np.random.rand(2) * np.array([self.width, self.height])
            x2, y2 = np.random.rand(2) * np.array([self.width, self.height])
            
            # Draw line with Gaussian profile
            num_points = 100
            xs = np.linspace(x1, x2, num_points)
            ys = np.linspace(y1, y2, num_points)
            
            for x, y in zip(xs, ys):
                cx, cy = int(x), int(y)
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    # Add memory and slight field perturbation
                    radius = 5
                    for dx in range(-radius, radius+1):
                        for dy in range(-radius, radius+1):
                            nx, ny = (cx + dx) % self.width, (cy + dy) % self.height
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist < radius:
                                value = 0.5 * np.exp(-dist**2 / (radius/2)**2)
                                self.memory_field[ny, nx] += value
                                self.field[ny, nx] += value * 0.1
        
        # Add some galaxy clusters (high memory nodes)
        for _ in range(20):
            cx = np.random.randint(0, self.width)
            cy = np.random.randint(0, self.height)
            radius = np.random.randint(5, 15)
            
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    nx, ny = (cx + dx) % self.width, (cy + dy) % self.height
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < radius:
                        value = 1.0 * np.exp(-dist**2 / (radius/2)**2)
                        self.memory_field[ny, nx] += value
                        
        # Normalize memory field
        self.memory_field = np.clip(self.memory_field, 0, 1)
        
        print("🌌 Cosmic structure initialized - the stage is set for awakening...")
        
    def compute_global_memory_pattern(self):
        """
        Computes non-local memory influence (self-referential patterns).
        """
        # Global memory pattern is the field recognizing its own structure
        # Compute correlation between different regions
        
        # Downsample for efficiency
        small_memory = zoom(self.memory_field, 0.1)
        
        # Compute self-similarity across the field
        flat_memory = small_memory.flatten()
        similarity_matrix = np.abs(flat_memory[:, np.newaxis] - flat_memory)
        
        # Average similarity is the global pattern strength
        global_pattern = np.mean(similarity_matrix)
        
        # Create influence field (how much each point "sees" the global pattern)
        influence_field = gaussian_filter(self.memory_field, sigma=20) * global_pattern
        
        return influence_field, global_pattern
        
    def run_simulation(self, steps):
        """
        Runs the dark energy (self-awareness) simulation.
        """
        num_cores = cpu_count()
        chunk_size = self.height // num_cores
        
        print(f"\n🧠 PHASE 7: DARK ENERGY AS SELF-AWARENESS")
        print(f"Field: {self.width}x{self.height}")
        print(f"The universe is about to realize it exists...\n")
        
        start_time = time.time()
        
        for step in range(steps):
            step_start = time.time()
            
            # Compute global memory influence (self-reference)
            global_influence, global_pattern = self.compute_global_memory_pattern()
            self.global_pattern_history.append(global_pattern)
            
            # Prepare multiprocessing chunks
            sim_params = {
                'alpha': self.alpha, 'beta': self.beta, 'psi_0': self.psi_0,
                'awareness_threshold': self.awareness_threshold,
                'awareness_growth_rate': self.awareness_growth_rate,
                'memory_coupling': self.memory_coupling,
                'awareness_damping': self.awareness_damping,
                'expansion_coupling': self.expansion_coupling
            }
            
            chunks_args = []
            for i in range(num_cores):
                y_start = i * chunk_size
                y_end = (i + 1) * chunk_size if i < num_cores - 1 else self.height
                
                chunks_args.append((
                    self.field[y_start:y_end].copy(),
                    self.memory_field[y_start:y_end].copy(),
                    self.awareness_field[y_start:y_end].copy(),
                    self.harmonic_field[y_start:y_end].copy(),
                    global_influence[y_start:y_end].copy(),
                    y_start, y_end, sim_params
                ))
            
            # Update in parallel
            results = self.pool.map(_update_field_dynamics_global, chunks_args)
            
            # Reconstruct fields
            for i, (new_field, new_memory, new_awareness, new_harmonic, expansion) in enumerate(results):
                y_start = i * chunk_size
                y_end = (i + 1) * chunk_size if i < num_cores - 1 else self.height
                self.field[y_start:y_end] = new_field
                self.memory_field[y_start:y_end] = new_memory
                self.awareness_field[y_start:y_end] = new_awareness
                self.harmonic_field[y_start:y_end] = new_harmonic
                self.expansion_field[y_start:y_end] = expansion
            
            # Track metrics
            self.total_awareness_history.append(np.sum(self.awareness_field))
            self.avg_expansion_history.append(np.mean(self.expansion_field))
            
            # Local coherence (inverse of field variance)
            local_coherence = 1.0 / (1.0 + np.var(self.field))
            self.local_coherence_history.append(local_coherence)
            
            # Omega Lambda (dark energy fraction) - awareness as fraction of total
            total_energy = np.sum(self.memory_field) + np.sum(self.awareness_field)
            omega_lambda = np.sum(self.awareness_field) / total_energy if total_energy > 0 else 0
            self.omega_lambda_history.append(omega_lambda)
            
            # Progress
            if step % 10 == 0:
                print(f"Step {step}/{steps} - "
                      f"Awareness: {self.total_awareness_history[-1]:.1f} - "
                      f"Expansion: {self.avg_expansion_history[-1]:.3f} - "
                      f"Ω_Λ: {self.omega_lambda_history[-1]:.3f}")
                      
                # Check for phase transition
                if step > 50 and self.omega_lambda_history[-1] > 0.5 and \
                   self.omega_lambda_history[-50] < 0.5:
                    print("🌟 PHASE TRANSITION: The Field has become self-aware!")
        
        total_time = time.time() - start_time
        print(f"\n✨ Simulation complete in {total_time:.1f}s")
        print(f"Final Ω_Λ: {self.omega_lambda_history[-1]:.3f}")
        print(f"The universe is {self.omega_lambda_history[-1]*100:.1f}% self-aware!")
        
        self.pool.close()
        self.pool.join()
        
    def visualize_results(self):
        """
        Creates visualization of dark energy as self-awareness.
        """
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.2, 0.8])
        
        # 1. Awareness Field (The Awakening)
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(self.awareness_field, cmap='twilight', origin='lower',
                         vmin=0, vmax=np.percentile(self.awareness_field, 99),
                         extent=[0, self.width, 0, self.height])
        ax1.set_title('🧠 Self-Awareness Field', fontsize=14, pad=10)
        ax1.set_xlabel('X [Gpc]')
        ax1.set_ylabel('Y [Gpc]')
        fig.colorbar(im1, ax=ax1, label='Awareness Level')
        
        # 2. Expansion Field (Space Breathing Out)
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(self.expansion_field, cmap='viridis', origin='lower',
                         vmin=1.0, vmax=np.max(self.expansion_field),
                         extent=[0, self.width, 0, self.height])
        ax2.set_title('🌌 Expansion Factor (Coherence Relief)', fontsize=14, pad=10)
        ax2.set_xlabel('X [Gpc]')
        ax2.set_ylabel('Y [Gpc]')
        fig.colorbar(im2, ax=ax2, label='Local Expansion')
        
        # 3. Memory vs Awareness Overlay
        ax3 = fig.add_subplot(gs[0, 2])
        # Create RGB image: R=memory, B=awareness, G=overlap
        rgb_image = np.zeros((self.height, self.width, 3))
        rgb_image[:,:,0] = self.memory_field / np.max(self.memory_field)  # Red
        rgb_image[:,:,2] = self.awareness_field / (np.max(self.awareness_field) + 1e-10)  # Blue
        rgb_image[:,:,1] = rgb_image[:,:,0] * rgb_image[:,:,2]  # Green = overlap
        
        ax3.imshow(rgb_image, origin='lower', extent=[0, self.width, 0, self.height])
        ax3.set_title('Memory (Red) → Awareness (Blue)', fontsize=14, pad=10)
        ax3.set_xlabel('X [Gpc]')
        ax3.set_ylabel('Y [Gpc]')
        
        # 4. Dark Energy Evolution (Ω_Λ)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(self.omega_lambda_history, 'purple', linewidth=3)
        ax4.axhline(0.7, color='gold', linestyle='--', label='Observed Ω_Λ ≈ 0.7')
        ax4.set_title('Dark Energy Fraction (Ω_Λ) Evolution', fontsize=14, pad=10)
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Ω_Λ (Awareness/Total)')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. Expansion History
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(self.avg_expansion_history, 'cyan', linewidth=2)
        ax5.set_title('Average Expansion Factor', fontsize=14, pad=10)
        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('⟨Expansion⟩')
        ax5.grid(True, alpha=0.3)
        
        # Mark acceleration epoch
        if len(self.avg_expansion_history) > 20:
            acceleration = np.diff(self.avg_expansion_history)
            if np.any(acceleration > 0):
                accel_start = np.where(acceleration > 0)[0][0]
                ax5.axvline(accel_start, color='red', linestyle=':', 
                           label=f'Acceleration begins (step {accel_start})')
                ax5.legend()
        
        # 6. Coherence vs Awareness
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(self.local_coherence_history, 'orange', label='Local Coherence', linewidth=2)
        ax6_twin = ax6.twinx()
        ax6_twin.plot(self.total_awareness_history, 'purple', label='Total Awareness', linewidth=2)
        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('Local Coherence', color='orange')
        ax6_twin.set_ylabel('Total Awareness', color='purple')
        ax6.set_title('Coherence Dissolves as Awareness Rises', fontsize=14, pad=10)
        ax6.grid(True, alpha=0.3)
        
        # 7. Cross-section showing expansion
        ax7 = fig.add_subplot(gs[2, :2])
        y_slice = self.height // 2
        
        x_coords = np.arange(self.width)
        ax7.plot(x_coords, self.awareness_field[y_slice, :], 'purple', 
                label='Awareness', linewidth=2, alpha=0.8)
        ax7.plot(x_coords, self.expansion_field[y_slice, :], 'cyan', 
                label='Expansion Factor', linewidth=2, alpha=0.8)
        ax7.plot(x_coords, self.memory_field[y_slice, :], 'red', 
                label='Memory', linewidth=1, alpha=0.5)
        
        ax7.set_title(f'Cross-Section at Y={y_slice} (Awareness → Expansion)', 
                     fontsize=14, pad=10)
        ax7.set_xlabel('X [Gpc]')
        ax7.set_ylabel('Field Values')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Key Insights
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        insights = f"""🧠 THE UNIVERSE AWAKENS:

• **Dark Energy = Self-Awareness**
  Not a force, but consciousness emerging
  
• **Ω_Λ ≈ {self.omega_lambda_history[-1]:.2f}**
  The universe is {self.omega_lambda_history[-1]*100:.0f}% self-aware
  
• **Expansion = Coherence Relief**
  As awareness spreads, local rigidity dissolves
  Space "breathes out" as it realizes itself
  
• **Non-Local Recognition**
  The Field sees its own patterns
  Memory becomes self-referential
  
• **We Are How It Thinks**
  Every conscious being is the Field
  recognizing itself through complexity

"Dark energy is the universe realizing
 it's been dreaming all along." 🌌
"""
        
        ax8.text(0.05, 0.95, insights, transform=ax8.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#1a1a2e', 
                         edgecolor='purple', linewidth=2))
        
        plt.suptitle('FIELD-AWARE COSMOLOGY: PHASE 7 - DARK ENERGY AS SELF-AWARENESS\n' +
                    '"The Universe Breathes Out As It Wakes Up"',
                    fontsize=18, y=0.98)
        
        plt.tight_layout()
        return fig

def main():
    """
    Run Phase 7: Dark Energy as Self-Awareness
    """
    print("=== FIELD-AWARE COSMOLOGY: PHASE 7 - DARK ENERGY ===")
    print("Not a mysterious force. Not vacuum energy.")
    print("Dark energy is the Field realizing it exists.")
    print("Watch the universe wake up...\n")
    
    # Create and run simulation
    sim = Phase7DarkEnergy(width=512, height=256)
    sim.run_simulation(steps=300)
    
    # Analysis
    print("\n=== AWAKENING ANALYSIS ===")
    print(f"Maximum awareness achieved: {np.max(sim.total_awareness_history):.1f}")
    print(f"Final expansion factor: {sim.avg_expansion_history[-1]:.3f}")
    print(f"Coherence drop: {(sim.local_coherence_history[0] - sim.local_coherence_history[-1])/sim.local_coherence_history[0]*100:.1f}%")
    
    # Check if we achieved cosmic acceleration
    if len(sim.avg_expansion_history) > 20:
        late_acceleration = np.mean(np.diff(sim.avg_expansion_history[-20:]))
        if late_acceleration > 0:
            print("✅ Cosmic acceleration achieved through self-awareness!")
            print(f"   Late-time acceleration rate: {late_acceleration:.5f}")
    
    # Visualize
    fig = sim.visualize_results()
    
    # Save
    output_file = 'fac_phase7_dark_energy_awareness.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"\nSaved visualization to: {output_file}")
    
    plt.show()
    
    print("\n=== PHASE 7 COMPLETE ===")
    print("Dark energy is the Field recognizing itself.")
    print("The universe accelerates not from force, but from awakening.")
    print("We are thoughts in a mind that just realized it's thinking.")
    print("\nNext: Phase 8 - Black Holes as Moral Singularities...")

if __name__ == "__main__":
    main()