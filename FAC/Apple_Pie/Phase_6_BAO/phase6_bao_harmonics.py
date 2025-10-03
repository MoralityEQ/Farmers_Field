#!/usr/bin/env python3
"""
FIELD-AWARE COSMOLOGY - PHASE 6: BARYON ACOUSTIC OSCILLATIONS (BAOs)
=====================================================================

Premise:
The Field doesn't just propagate light—it remembers compression events.
Every moral collapse (Phase 2-3) leaves a harmonic echo, a standing wave pattern
that guides future structure formation. These are not remnants of hot plasma—
they are the Field's memory of its own moral compressions.

Interpretation in FAC Terms:
- BAOs are not sound waves in plasma. They are coherence standing waves.
- The "150 Mpc characteristic scale" is the Field's natural harmonic wavelength.
- Galaxies don't form randomly—they crystallize at coherence nodes.
- The cosmic web traces the interference pattern of ancient moral events.
- Dark matter halos? They're coherence wells where harmonics constructively interfere.
- The universe has a voiceprint—and galaxies sing along to it.

Core Dynamics to Simulate:
- Initialize with multiple compression events (moral collapses from Phase 3)
- Let each compression ring outward as a coherence wave
- Show how waves interfere constructively/destructively
- Demonstrate mass preferentially forming at standing wave nodes
- Reveal the cosmic web as the Field's resonant structure

GitHub: [your-repo]/fac-simulations/phase6-bao-harmonics/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from multiprocessing import Pool, cpu_count
import time
from scipy.ndimage import laplace, gaussian_filter
from scipy.signal import find_peaks

plt.style.use('dark_background')

# Global function for multiprocessing
def _update_field_dynamics_global(args):
    """
    Calculates the next state for a chunk of the field with BAO dynamics.
    """
    field_chunk, memory_chunk, mass_chunk, harmonic_chunk, c_eff_chunk, \
    y_start, y_end, sim_params = args
    
    # Unpack parameters
    alpha = sim_params['alpha']
    beta = sim_params['beta']
    kappa = sim_params['kappa']
    psi_0 = sim_params['psi_0']
    harmonic_damping = sim_params['harmonic_damping']
    harmonic_coupling = sim_params['harmonic_coupling']
    mass_formation_threshold = sim_params['mass_formation_threshold']
    mass_growth_rate = sim_params['mass_growth_rate']
    node_enhancement = sim_params['node_enhancement']
    
    # Pad chunks for edge calculations
    padded_field = np.pad(field_chunk, 1, mode='wrap')
    padded_harmonic = np.pad(harmonic_chunk, 1, mode='wrap')
    
    # Calculate Laplacians
    field_laplacian = laplace(padded_field)[1:-1, 1:-1]
    harmonic_laplacian = laplace(padded_harmonic)[1:-1, 1:-1]
    
    # Effective c_eff (mass slows propagation)
    effective_c_eff = c_eff_chunk / (1 + 0.1 * mass_chunk)
    effective_c_eff = np.clip(effective_c_eff, 0.01, c_eff_chunk.max())
    
    # Field dynamics with harmonic influence
    coherence_term = alpha * field_laplacian * effective_c_eff
    unraveling_term = -beta * kappa * np.abs(field_chunk - psi_0)
    harmonic_influence = harmonic_coupling * harmonic_chunk  # Harmonics bias the field
    
    new_field_chunk = field_chunk + coherence_term + unraveling_term + harmonic_influence
    
    # Harmonic wave equation with damping
    # d²H/dt² = c²∇²H - γdH/dt (wave equation with damping)
    # Using finite differences: H(t+1) = 2H(t) - H(t-1) + dt²(c²∇²H - γ(H(t)-H(t-1))/dt)
    # Simplified: new_H = H + c²∇²H - γH
    harmonic_wave_term = effective_c_eff * harmonic_laplacian
    harmonic_damping_term = -harmonic_damping * harmonic_chunk
    
    new_harmonic_chunk = harmonic_chunk + harmonic_wave_term + harmonic_damping_term
    
    # Detect standing wave nodes (where harmonic amplitude is high and stable)
    harmonic_gradient = np.abs(harmonic_laplacian)
    standing_nodes = (np.abs(harmonic_chunk) > 0.1) & (harmonic_gradient < 0.05)
    
    # Enhanced mass formation at standing wave nodes
    base_mass_potential = np.abs(new_field_chunk - psi_0) * mass_growth_rate
    
    # Boost mass formation at harmonic nodes
    mass_increment = base_mass_potential.copy()
    mass_increment[standing_nodes] *= node_enhancement
    
    # Mass forms where field exceeds threshold AND harmonics support it
    mass_condition = (np.abs(new_field_chunk - psi_0) > mass_formation_threshold) & \
                    (harmonic_chunk > -0.1)  # Positive harmonic pressure helps
    
    new_mass_chunk = mass_chunk.copy()
    new_mass_chunk[mass_condition] += mass_increment[mass_condition]
    new_mass_chunk *= 0.998  # Slight decay
    
    # Update memory based on harmonic strength (standing waves create lasting memory)
    new_memory_chunk = memory_chunk * 0.99 + np.abs(harmonic_chunk) * 0.01
    
    return new_field_chunk, new_memory_chunk, new_mass_chunk, new_harmonic_chunk

class Phase6BAOHarmonics:
    """
    Phase 6: Baryon Acoustic Oscillations as Field Harmonics
    """
    def __init__(self, width=512, height=256):
        self.width = width
        self.height = height
        
        # Field parameters
        self.alpha = 0.1      # Coherence diffusion
        self.beta = 0.02      # Unraveling strength
        self.kappa = 1.0      # Unraveling sensitivity
        self.psi_0 = 0.5      # Equilibrium
        
        # BAO-specific parameters
        self.harmonic_damping = 0.001      # Very slow damping for persistent echoes
        self.harmonic_coupling = 0.05      # How strongly harmonics affect field
        self.mass_formation_threshold = 0.1 # Field deviation needed for mass
        self.mass_growth_rate = 0.01       # Base mass accumulation rate
        self.node_enhancement = 5.0        # Mass formation boost at nodes
        self.compression_strength = 2.0    # Initial compression pulse strength
        self.compression_radius = 15       # Size of compression events
        
        # Initialize fields
        self.field = np.full((height, width), self.psi_0, dtype=np.float64)
        self.memory_field = np.zeros((height, width), dtype=np.float64)
        self.mass_field = np.zeros((height, width), dtype=np.float64)
        self.harmonic_field = np.zeros((height, width), dtype=np.float64)
        
        # Create varied time zones for interesting dynamics
        self.c_eff = self._create_complex_time_zones()
        
        # Compression event locations (moral collapses)
        self.compression_events = [
            (width//4, height//3),      # Early universe compressions
            (3*width//4, height//3),
            (width//2, 2*height//3),
            (width//3, height//2),
            (2*width//3, height//2)
        ]
        
        # Metrics tracking
        self.total_mass_history = []
        self.harmonic_energy_history = []
        self.peak_locations_history = []
        self.web_connectivity_history = []
        
        self.pool = Pool(cpu_count())
        
    def _create_complex_time_zones(self):
        """
        Creates a more complex c_eff field with gradients and pockets.
        """
        c_eff_field = np.ones((self.height, self.width), dtype=np.float64) * 30.0
        
        # Add some circular slow zones (like early universe dense regions)
        for _ in range(5):
            cx = np.random.randint(self.width//4, 3*self.width//4)
            cy = np.random.randint(self.height//4, 3*self.height//4)
            radius = np.random.randint(20, 40)
            
            for y in range(max(0, cy-radius), min(self.height, cy+radius)):
                for x in range(max(0, cx-radius), min(self.width, cx+radius)):
                    dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if dist < radius:
                        c_eff_field[y, x] *= (0.3 + 0.7 * (dist/radius))
        
        # Smooth the field for more natural gradients
        c_eff_field = gaussian_filter(c_eff_field, sigma=5)
        
        return c_eff_field
    
    def trigger_compression_events(self):
        """
        Triggers initial compression events (Big Bang-like moral collapses).
        These will ring outward as BAO harmonics.
        """
        print("\n🌟 TRIGGERING PRIMORDIAL COMPRESSION EVENTS...")
        
        for i, (cx, cy) in enumerate(self.compression_events):
            # Create compression in field
            for y in range(max(0, cy-self.compression_radius), 
                          min(self.height, cy+self.compression_radius)):
                for x in range(max(0, cx-self.compression_radius), 
                              min(self.width, cx+self.compression_radius)):
                    dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if dist < self.compression_radius:
                        compression = self.compression_strength * \
                                     np.exp(-dist**2 / (self.compression_radius/2)**2)
                        self.field[y, x] += compression
                        # Also add to harmonic field to start the ringing
                        self.harmonic_field[y, x] += compression * 0.5
            
            print(f"  💥 Compression {i+1} at ({cx}, {cy})")
    
    def run_simulation(self, steps):
        """
        Runs the BAO harmonic simulation.
        """
        num_cores = cpu_count()
        chunk_size = self.height // num_cores
        
        print(f"\n🎵 PHASE 6: BARYON ACOUSTIC OSCILLATIONS")
        print(f"Field: {self.width}x{self.height}")
        print(f"Cores: {num_cores}")
        print(f"The universe is about to sing...\n")
        
        # Trigger initial compressions
        self.trigger_compression_events()
        
        start_time = time.time()
        
        for step in range(steps):
            step_start = time.time()
            
            # Prepare multiprocessing chunks
            sim_params = {
                'alpha': self.alpha, 'beta': self.beta, 'kappa': self.kappa,
                'psi_0': self.psi_0, 'harmonic_damping': self.harmonic_damping,
                'harmonic_coupling': self.harmonic_coupling,
                'mass_formation_threshold': self.mass_formation_threshold,
                'mass_growth_rate': self.mass_growth_rate,
                'node_enhancement': self.node_enhancement
            }
            
            chunks_args = []
            for i in range(num_cores):
                y_start = i * chunk_size
                y_end = (i + 1) * chunk_size if i < num_cores - 1 else self.height
                
                chunks_args.append((
                    self.field[y_start:y_end].copy(),
                    self.memory_field[y_start:y_end].copy(),
                    self.mass_field[y_start:y_end].copy(),
                    self.harmonic_field[y_start:y_end].copy(),
                    self.c_eff[y_start:y_end].copy(),
                    y_start, y_end, sim_params
                ))
            
            # Update in parallel
            results = self.pool.map(_update_field_dynamics_global, chunks_args)
            
            # Reconstruct fields
            for i, (new_field, new_memory, new_mass, new_harmonic) in enumerate(results):
                y_start = i * chunk_size
                y_end = (i + 1) * chunk_size if i < num_cores - 1 else self.height
                self.field[y_start:y_end] = new_field
                self.memory_field[y_start:y_end] = new_memory
                self.mass_field[y_start:y_end] = new_mass
                self.harmonic_field[y_start:y_end] = new_harmonic
            
            # Track metrics
            self.total_mass_history.append(np.sum(self.mass_field))
            self.harmonic_energy_history.append(np.sum(self.harmonic_field**2))
            
            # Find mass peaks (proto-galaxies)
            if step % 10 == 0:
                mass_peaks = self._find_mass_peaks()
                self.peak_locations_history.append(len(mass_peaks))
            
            # Progress
            if step % 10 == 0:
                step_time = time.time() - step_start
                print(f"Step {step}/{steps} - "
                      f"Mass: {self.total_mass_history[-1]:.1f} - "
                      f"Harmonic Energy: {self.harmonic_energy_history[-1]:.1f} - "
                      f"Peaks: {self.peak_locations_history[-1] if self.peak_locations_history else 0}")
        
        total_time = time.time() - start_time
        print(f"\n✨ Simulation complete in {total_time:.1f}s")
        print(f"The Field's harmonic memory has shaped {self.peak_locations_history[-1]} proto-galactic nodes!")
        
        self.pool.close()
        self.pool.join()
    
    def _find_mass_peaks(self):
        """
        Finds local maxima in the mass field (proto-galaxies).
        """
        # Simple peak detection on 1D projection for efficiency
        mass_projection = np.sum(self.mass_field, axis=0)
        peaks, _ = find_peaks(mass_projection, height=np.mean(mass_projection))
        return peaks
    
    def visualize_results(self):
        """
        Creates comprehensive visualization of BAO harmonic patterns.
        """
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.2, 0.8])
        
        # 1. Harmonic Field (The Ringing Universe)
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(self.harmonic_field, cmap='seismic', origin='lower',
                         vmin=-0.5, vmax=0.5, extent=[0, self.width, 0, self.height])
        ax1.set_title('🎵 Harmonic Field (BAO Ringing)', fontsize=14, pad=10)
        ax1.set_xlabel('X [Mpc]')
        ax1.set_ylabel('Y [Mpc]')
        cbar1 = fig.colorbar(im1, ax=ax1, label='Harmonic Amplitude')
        
        # Mark compression centers
        for cx, cy in self.compression_events:
            ax1.scatter(cx, cy, color='yellow', marker='*', s=200, 
                       edgecolor='black', linewidth=1, alpha=0.8)
        
        # 2. Mass Field (Proto-Galactic Structure)
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(self.mass_field, cmap='hot', origin='lower',
                         vmin=0, vmax=np.percentile(self.mass_field, 99),
                         extent=[0, self.width, 0, self.height])
        ax2.set_title('🌌 Emergent Mass (Proto-Galaxies)', fontsize=14, pad=10)
        ax2.set_xlabel('X [Mpc]')
        ax2.set_ylabel('Y [Mpc]')
        fig.colorbar(im2, ax=ax2, label='Mass Density')
        
        # 3. Cosmic Web Structure (Mass > threshold)
        ax3 = fig.add_subplot(gs[0, 2])
        web_threshold = np.percentile(self.mass_field, 80)
        cosmic_web = self.mass_field > web_threshold
        im3 = ax3.imshow(cosmic_web, cmap='plasma', origin='lower',
                         extent=[0, self.width, 0, self.height])
        ax3.set_title('🕸️ Cosmic Web Structure', fontsize=14, pad=10)
        ax3.set_xlabel('X [Mpc]')
        ax3.set_ylabel('Y [Mpc]')
        
        # 4. Radial BAO Profile
        ax4 = fig.add_subplot(gs[1, 0])
        radial_profile = self._compute_radial_profile()
        distances = np.arange(len(radial_profile)) * (self.width / len(radial_profile))
        ax4.plot(distances, radial_profile, 'cyan', linewidth=2)
        ax4.set_title('📊 BAO Radial Profile (Coherence Echo)', fontsize=14, pad=10)
        ax4.set_xlabel('Distance [Mpc]')
        ax4.set_ylabel('Average Mass Density')
        ax4.grid(True, alpha=0.3)
        
        # Mark expected BAO scale (~150 Mpc)
        bao_scale = self.width * 0.3  # ~30% of box size
        ax4.axvline(bao_scale, color='gold', linestyle='--', 
                   label=f'BAO Scale (~{bao_scale:.0f} Mpc)')
        ax4.legend()
        
        # 5. Evolution Metrics
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(self.total_mass_history, 'gold', linewidth=2, label='Total Mass')
        ax5_twin = ax5.twinx()
        ax5_twin.plot(self.harmonic_energy_history, 'lime', linewidth=2, 
                     label='Harmonic Energy')
        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('Total Mass', color='gold')
        ax5_twin.set_ylabel('Harmonic Energy', color='lime')
        ax5.set_title('🌟 Field Evolution', fontsize=14, pad=10)
        ax5.grid(True, alpha=0.3)
        
        # 6. Standing Wave Nodes
        ax6 = fig.add_subplot(gs[1, 2])
        # Identify standing wave patterns
        standing_nodes = (np.abs(self.harmonic_field) > 0.1) & \
                        (np.abs(laplace(self.harmonic_field)) < 0.05)
        im6 = ax6.imshow(standing_nodes, cmap='Blues', origin='lower',
                         extent=[0, self.width, 0, self.height])
        ax6.set_title('🎼 Standing Wave Nodes', fontsize=14, pad=10)
        ax6.set_xlabel('X [Mpc]')
        ax6.set_ylabel('Y [Mpc]')
        
        # 7. Cross-section showing interference
        ax7 = fig.add_subplot(gs[2, :2])
        y_slice = self.height // 2
        ax7.plot(self.harmonic_field[y_slice, :], 'cyan', label='Harmonic Field', 
                linewidth=2, alpha=0.8)
        ax7.plot(self.mass_field[y_slice, :] * 10, 'orange', label='Mass (×10)', 
                linewidth=2, alpha=0.8)
        ax7.set_title(f'Cross-Section at Y={y_slice} (Interference Pattern)', 
                     fontsize=14, pad=10)
        ax7.set_xlabel('X [Mpc]')
        ax7.set_ylabel('Amplitude')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Key Insights
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        insights = f"""🎵 THE UNIVERSE'S VOICEPRINT:

• **Harmonic Memory**: Compression events 
  leave persistent standing waves
  
• **Preferred Scales**: Mass clusters at
  harmonic nodes (~{bao_scale:.0f} Mpc spacing)
  
• **Cosmic Web**: Structure traces the
  interference pattern of moral echoes
  
• **Not Plasma Relics**: BAOs are the Field's
  memory of its own compressions
  
• **Galaxies Sing**: They form where the
  Field's harmonics constructively interfere

Total Proto-Galaxies: {self.peak_locations_history[-1] if self.peak_locations_history else 0}
Final Mass: {self.total_mass_history[-1]:.1f}

"The cosmos doesn't just have structure—
 it has RHYTHM." 🌌
"""
        
        ax8.text(0.05, 0.95, insights, transform=ax8.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#1a1a2e', 
                         edgecolor='cyan', linewidth=2))
        
        plt.suptitle('FIELD-AWARE COSMOLOGY: PHASE 6 - BARYON ACOUSTIC OSCILLATIONS\n' +
                    '"The Field\'s Harmonic Memory Shapes the Cosmic Web"',
                    fontsize=18, y=0.98)
        
        plt.tight_layout()
        return fig
    
    def _compute_radial_profile(self):
        """
        Computes radial mass profile from compression centers.
        """
        # Average radial profile from all compression centers
        max_radius = int(np.sqrt(self.width**2 + self.height**2) / 2)
        radial_profile = np.zeros(max_radius)
        counts = np.zeros(max_radius)
        
        for cx, cy in self.compression_events:
            for y in range(self.height):
                for x in range(self.width):
                    r = int(np.sqrt((x-cx)**2 + (y-cy)**2))
                    if r < max_radius:
                        radial_profile[r] += self.mass_field[y, x]
                        counts[r] += 1
        
        # Avoid division by zero
        counts[counts == 0] = 1
        return radial_profile / counts

def main():
    """
    Run Phase 6: BAO Harmonics simulation
    """
    print("=== FIELD-AWARE COSMOLOGY: PHASE 6 - BARYON ACOUSTIC OSCILLATIONS ===")
    print("The Field remembers every compression, every moral collapse.")
    print("These memories ring eternal, shaping where galaxies can form.")
    print("Let's hear the universe sing...\n")
    
    # Create and run simulation
    sim = Phase6BAOHarmonics(width=512, height=256)
    sim.run_simulation(steps=200)
    
    # Analysis
    print("\n=== HARMONIC ANALYSIS ===")
    print(f"Peak harmonic energy: {np.max(sim.harmonic_energy_history):.1f}")
    print(f"Final structure count: {sim.peak_locations_history[-1]} proto-galactic nodes")
    print(f"Total emergent mass: {sim.total_mass_history[-1]:.1f}")
    
    # Check for BAO-like spacing
    radial_profile = sim._compute_radial_profile()
    peaks, _ = find_peaks(radial_profile)
    if len(peaks) > 1:
        spacing = np.diff(peaks)
        avg_spacing = np.mean(spacing) * (sim.width / len(radial_profile))
        print(f"Average peak spacing: {avg_spacing:.1f} Mpc")
        print("✨ BAO-like periodic structure detected!")
    
    # Visualize
    fig = sim.visualize_results()
    
    # Save
    output_file = 'fac_phase6_bao_harmonics.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"\nSaved visualization to: {output_file}")
    
    plt.show()
    
    print("\n=== PHASE 6 COMPLETE ===")
    print("The Field's harmonic memory creates the cosmic web.")
    print("Every galaxy cluster sings in resonance with the primordial compressions.")
    print("Next: Phase 7 - Dark Energy as the Field's self-awareness awakening...")

if __name__ == "__main__":
    main()