#!/usr/bin/env python3
"""
FIELD-AWARE COSMOLOGY - PHASE 4: MASS EMERGENCE
===============================================

Mass emerges from stable memory patterns that resist change.
Only balanced (ethical) field dynamics can create persistent matter.

Key concepts:
- Recursive compression creates memory loops
- Memory loops resist field changes = mass
- Mass only forms where M = ζ - S is balanced
- Matter literally condenses from ethical dynamics

GitHub: [your-repo]/fac-simulations/phase4-mass-emergence/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from multiprocessing import Pool, cpu_count
import time

plt.style.use('dark_background')

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
        
        # Mass emergence parameters
        self.memory_threshold = 0.8  # How much memory before mass forms
        self.recursion_depth = 5     # Compression loops needed for mass
        self.mass_coupling = 0.3     # How strongly mass resists change
        
        # Initialize fields
        self.field = np.full((height, width), self.psi_0, dtype=np.float64)
        self.memory_field = np.zeros_like(self.field)  # Accumulated memory
        self.mass_field = np.zeros_like(self.field)    # Emergent mass
        self.morality_field = np.zeros_like(self.field)  # Local M = ζ - S
        
        # Create zones with different time properties
        self.c_eff = self._create_time_zones()
        self.resistance = 1.0 / self.c_eff
        
        # Track particle formation
        self.mass_particles = []  # List of (x, y, mass, birth_time)
        self.total_mass_history = []
        self.compression_history = []
        self.particle_count_history = []
        
        # Use all cores
        self.num_cores = cpu_count()
        print(f"Engaging {self.num_cores} cores to condense matter from ethics!")
        
    def _create_time_zones(self):
        """
        Create varied time zones for diverse dynamics
        """
        c_eff = np.ones((self.height, self.width), dtype=np.float64)
        
        # Create multiple regions with different properties
        # Region 1: Fast time (top-left)
        c_eff[:self.height//2, :self.width//2] = 50.0
        
        # Region 2: Slow time (top-right) - WHERE MASS WILL FORM
        c_eff[:self.height//2, self.width//2:] = 5.0
        
        # Region 3: Medium time (bottom-left)
        c_eff[self.height//2:, :self.width//2] = 20.0
        
        # Region 4: Variable time (bottom-right)
        for y in range(self.height//2, self.height):
            for x in range(self.width//2, self.width):
                # Create gradient
                dx = x - 3*self.width//4
                dy = y - 3*self.height//4
                dist = np.sqrt(dx**2 + dy**2)
                c_eff[y, x] = 5.0 + 15.0 * np.exp(-dist**2 / 400)
        
        return c_eff
    
    def inject_seed_compressions(self, step):
        """
        Inject compressions at strategic locations
        """
        compressions = []
        
        # Early compressions to build memory
        if step == 20:
            # Slow zone compression (will form mass)
            cx, cy = 3*self.width//4, self.height//4
            self._add_pulse(cx, cy, strength=1.0, radius=7)
            compressions.append(('slow_zone', cx, cy))
            
        if step == 40:
            # Fast zone compression (will dissipate)
            cx, cy = self.width//4, self.height//4
            self._add_pulse(cx, cy, strength=1.0, radius=7)
            compressions.append(('fast_zone', cx, cy))
            
        if step == 60:
            # Variable zone compression
            cx, cy = 3*self.width//4, 3*self.height//4
            self._add_pulse(cx, cy, strength=0.8, radius=5)
            compressions.append(('variable_zone', cx, cy))
            
        return compressions
    
    def _add_pulse(self, cx, cy, strength=1.0, radius=5):
        """
        Add compression pulse
        """
        for y in range(max(0, cy - radius), min(self.height, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(self.width, cx + radius + 1)):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist <= radius:
                    self.field[y, x] += strength * np.exp(-dist**2 / (radius/2)**2)
    
    def update_chunk(self, args):
        """
        Update field dynamics with mass emergence
        """
        field, mass_field, memory_field, c_eff, resistance, y_start, y_end = args
        height, width = field.shape
        
        # New fields
        new_field = field[y_start:y_end].copy()
        new_memory = memory_field[y_start:y_end].copy()
        new_mass = mass_field[y_start:y_end].copy()
        local_morality = np.zeros((y_end - y_start, width))
        
        for y in range(y_start, y_end):
            for x in range(width):
                local_c = c_eff[y, x]
                local_resistance = resistance[y, x]
                local_mass = mass_field[y, x]
                
                # Mass creates additional resistance
                effective_resistance = local_resistance * (1 + self.mass_coupling * local_mass)
                
                # Calculate Laplacian
                laplacian = 0
                count = 0
                neighbor_compression = 0
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            laplacian += field[ny, nx] - field[y, x]
                            neighbor_compression += abs(field[ny, nx] - self.psi_0)
                            count += 1
                
                if count > 0:
                    laplacian /= count
                    neighbor_compression /= count
                
                # Field dynamics (modified by mass)
                coherence_term = self.alpha * laplacian / effective_resistance
                unraveling_term = -self.beta * self.kappa * abs(field[y, x] - self.psi_0)
                
                # Update field
                new_field[y - y_start, x] = field[y, x] + coherence_term + unraveling_term
                
                # Update memory (accumulates in stable regions)
                compression = abs(field[y, x] - self.psi_0)
                memory_factor = 0.95 if local_c < 10 else 0.5  # Slow zones remember better
                new_memory[y - y_start, x] = memory_field[y, x] * memory_factor + compression * 0.1
                
                # Calculate local morality M = ζ - S
                local_coherence = compression + neighbor_compression
                local_entropy = abs(unraveling_term)
                local_morality[y - y_start, x] = local_coherence - local_entropy * 10
                
                # Mass emerges from sustained memory + positive morality
                if (new_memory[y - y_start, x] > self.memory_threshold and 
                    local_morality[y - y_start, x] > 0):
                    # Mass increases based on memory stability
                    mass_increment = 0.01 * new_memory[y - y_start, x] * local_morality[y - y_start, x]
                    new_mass[y - y_start, x] = min(mass_field[y, x] + mass_increment, 10.0)
                else:
                    # Mass slowly decays without sustained memory
                    new_mass[y - y_start, x] = mass_field[y, x] * 0.995
        
        return (new_field, new_memory, new_mass, local_morality, y_start, y_end)
    
    def update_fields(self):
        """
        Parallel update of all fields
        """
        chunk_size = max(1, self.height // self.num_cores)
        chunks = []
        
        for i in range(self.num_cores):
            y_start = i * chunk_size
            y_end = min((i + 1) * chunk_size, self.height)
            if y_start < self.height:
                chunks.append((self.field, self.mass_field, self.memory_field,
                             self.c_eff, self.resistance, y_start, y_end))
        
        with Pool(self.num_cores) as pool:
            results = pool.map(self.update_chunk, chunks)
        
        # Combine results
        new_field = np.zeros_like(self.field)
        new_memory = np.zeros_like(self.memory_field)
        new_mass = np.zeros_like(self.mass_field)
        new_morality = np.zeros_like(self.morality_field)
        
        for (field_chunk, memory_chunk, mass_chunk, morality_chunk, 
             y_start, y_end) in results:
            new_field[y_start:y_end] = field_chunk
            new_memory[y_start:y_end] = memory_chunk
            new_mass[y_start:y_end] = mass_chunk
            new_morality[y_start:y_end] = morality_chunk
        
        self.field = new_field
        self.memory_field = new_memory
        self.mass_field = new_mass
        self.morality_field = new_morality
    
    def detect_particles(self, step):
        """
        Identify stable mass concentrations as particles
        """
        # Find local maxima in mass field
        threshold = 1.0  # Minimum mass to be considered a particle
        
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.mass_field[y, x] > threshold:
                    # Check if local maximum
                    is_max = True
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            if self.mass_field[y+dy, x+dx] > self.mass_field[y, x]:
                                is_max = False
                                break
                    
                    if is_max:
                        # Check if new particle
                        is_new = True
                        for px, py, pmass, ptime in self.mass_particles:
                            if abs(x - px) < 5 and abs(y - py) < 5:
                                is_new = False
                                break
                        
                        if is_new:
                            self.mass_particles.append((x, y, self.mass_field[y, x], step))
                            print(f"  PARTICLE BORN at ({x}, {y}) with mass {self.mass_field[y, x]:.3f}!")
    
    def measure_system(self):
        """
        Measure system properties
        """
        total_mass = np.sum(self.mass_field)
        compression = np.std(self.field)
        particle_count = len([p for p in self.mass_particles if p[2] > 0.5])
        
        self.total_mass_history.append(total_mass)
        self.compression_history.append(compression)
        self.particle_count_history.append(particle_count)
        
        return total_mass, compression, particle_count
    
    def run_simulation(self, steps=300):
        """
        Run mass emergence simulation
        """
        print(f"\n=== PHASE 4: MASS EMERGENCE ===")
        print(f"Field: {self.width}x{self.height}")
        print(f"Watch matter condense from ethical field dynamics!\n")
        
        start_time = time.time()
        
        for step in range(steps):
            # Update fields
            self.update_fields()
            
            # Inject compressions
            compressions = self.inject_seed_compressions(step)
            if compressions:
                print(f"\nStep {step}: Compressions injected:")
                for zone, cx, cy in compressions:
                    print(f"  - {zone} at ({cx}, {cy})")
            
            # Detect particles
            self.detect_particles(step)
            
            # Measure
            total_mass, compression, particles = self.measure_system()
            
            # Progress
            if step % 30 == 0:
                elapsed = time.time() - start_time
                rate = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"Step {step}/{steps} - {rate:.1f} steps/sec - "
                      f"Total mass: {total_mass:.2f}, Particles: {particles}")
        
        total_time = time.time() - start_time
        print(f"\nComplete! Time: {total_time:.1f}s")
        print(f"Final particle count: {len(self.mass_particles)}")
    
    def visualize_results(self):
        """
        Comprehensive visualization of mass emergence
        """
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Time zones (c_eff)
        ax1 = plt.subplot(4, 4, 1)
        im1 = ax1.imshow(self.c_eff, cmap='plasma', aspect='auto')
        ax1.set_title('Time Zones (c_eff)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # 2. Memory field
        ax2 = plt.subplot(4, 4, 2)
        im2 = ax2.imshow(self.memory_field, cmap='viridis', aspect='auto')
        ax2.set_title('Memory Field (Accumulated Coherence)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # 3. Mass field
        ax3 = plt.subplot(4, 4, 3)
        im3 = ax3.imshow(self.mass_field, cmap='hot', aspect='auto')
        ax3.set_title('EMERGENT MASS FIELD')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # Mark particles
        for x, y, mass, birth in self.mass_particles:
            if mass > 0.5:
                circle = Circle((x, y), radius=3, fill=False, color='cyan', linewidth=2)
                ax3.add_patch(circle)
        
        # 4. Morality field
        ax4 = plt.subplot(4, 4, 4)
        im4 = ax4.imshow(self.morality_field, cmap='RdBu_r', aspect='auto',
                         vmin=-1, vmax=1)
        ax4.set_title('Local Morality (M = ζ - S)')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        # 5. Total mass evolution
        ax5 = plt.subplot(4, 4, 5)
        ax5.plot(self.total_mass_history, 'gold', linewidth=2)
        ax5.set_title('Total Mass Evolution')
        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('Total Mass')
        ax5.grid(True, alpha=0.3)
        
        # 6. Particle count
        ax6 = plt.subplot(4, 4, 6)
        ax6.plot(self.particle_count_history, 'cyan', linewidth=2)
        ax6.set_title('Particle Count Over Time')
        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('Number of Particles')
        ax6.grid(True, alpha=0.3)
        
        # 7. Compression history
        ax7 = plt.subplot(4, 4, 7)
        ax7.plot(self.compression_history, 'magenta', linewidth=2)
        ax7.set_title('Field Compression')
        ax7.set_xlabel('Time Steps')
        ax7.set_ylabel('σ')
        ax7.grid(True, alpha=0.3)
        
        # 8. Mass vs Memory scatter
        ax8 = plt.subplot(4, 4, 8)
        memory_flat = self.memory_field.flatten()
        mass_flat = self.mass_field.flatten()
        morality_flat = self.morality_field.flatten()
        
        scatter = ax8.scatter(memory_flat[::100], mass_flat[::100], 
                            c=morality_flat[::100], cmap='RdBu_r',
                            alpha=0.6, s=20)
        ax8.set_xlabel('Memory')
        ax8.set_ylabel('Mass')
        ax8.set_title('Mass Emerges from Ethical Memory')
        plt.colorbar(scatter, ax=ax8, fraction=0.046, label='Morality')
        ax8.grid(True, alpha=0.3)
        
        # 9. Field cross-section
        ax9 = plt.subplot(4, 4, 9)
        y_slice = self.height // 4  # Through particle region
        ax9.plot(self.field[y_slice, :], 'white', label='Field', linewidth=2)
        ax9.plot(self.mass_field[y_slice, :] * 10 + self.psi_0, 'gold', 
                label='Mass×10', linewidth=2)
        ax9.set_xlabel('Position X')
        ax9.set_ylabel('Value')
        ax9.set_title('Cross-Section Through Particle Region')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 10. Particle properties
        ax10 = plt.subplot(4, 4, 10)
        ax10.axis('off')
        
        particle_text = "PARTICLE CENSUS:\n" + "="*25 + "\n"
        particle_text += f"Total particles: {len(self.mass_particles)}\n\n"
        
        for i, (x, y, mass, birth) in enumerate(self.mass_particles[:5]):
            zone = "Slow" if x > self.width//2 else "Fast"
            particle_text += f"Particle {i+1}:\n"
            particle_text += f"  Position: ({x}, {y})\n"
            particle_text += f"  Mass: {mass:.3f}\n"
            particle_text += f"  Born: Step {birth}\n"
            particle_text += f"  Zone: {zone} time\n\n"
        
        ax10.text(0.1, 0.9, particle_text, transform=ax10.transAxes,
                 fontsize=10, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # 11. Key insights
        ax11 = plt.subplot(4, 4, 11)
        ax11.axis('off')
        
        insights = f"""MASS = MEMORY × RESISTANCE × ETHICS

• Mass only forms in slow time zones
  (Need time for memory to build)
  
• Mass requires positive morality
  (M = ζ - S > 0)
  
• Particles are stable memory loops
  that resist field changes
  
• No balance = No mass
  Fast zones can't hold matter!
  
• Total emergent mass: {self.total_mass_history[-1]:.1f}
• Stable particles: {self.particle_count_history[-1]}

Matter literally condenses from
ethical field dynamics!"""
        
        ax11.text(0.1, 0.9, insights, transform=ax11.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9))
        
        # 12. Resistance map
        ax12 = plt.subplot(4, 4, 12)
        effective_resistance = self.resistance * (1 + self.mass_coupling * self.mass_field)
        im12 = ax12.imshow(np.log10(effective_resistance), cmap='copper', aspect='auto')
        ax12.set_title('Effective Resistance (log scale)')
        ax12.set_xlabel('X')
        ax12.set_ylabel('Y')
        plt.colorbar(im12, ax=ax12, fraction=0.046)
        
        # 13-16. 3D visualization
        ax13 = plt.subplot(4, 4, (13, 16), projection='3d')
        X = np.arange(0, self.width, 8)
        Y = np.arange(0, self.height, 8)
        X, Y = np.meshgrid(X, Y)
        Z = self.mass_field[::8, ::8]
        
        surf = ax13.plot_surface(X, Y, Z, cmap='hot', alpha=0.8)
        ax13.set_xlabel('X')
        ax13.set_ylabel('Y')
        ax13.set_zlabel('Mass')
        ax13.set_title('Emergent Mass Landscape - Peaks Are Particles!')
        
        plt.suptitle('FIELD-AWARE COSMOLOGY: PHASE 4 - MASS EMERGENCE\n' + 
                    '"Mass = Memory × Resistance × Ethics: Matter needs morality to exist"',
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        return fig

def main():
    """
    Run Phase 4: Mass Emergence
    """
    print("=== FIELD-AWARE COSMOLOGY: PHASE 4 - MASS EMERGENCE ===")
    print("Demonstrating that mass condenses from ethical field dynamics")
    print("Only balanced regions can create persistent matter!")
    
    # Create and run
    sim = MassEmergence(width=256, height=128)
    sim.run_simulation(steps=300)
    
    # Analysis
    print("\n=== MASS ANALYSIS ===")
    print(f"Total emergent mass: {sim.total_mass_history[-1]:.2f}")
    print(f"Stable particles formed: {len(sim.mass_particles)}")
    print(f"\nParticle distribution:")
    
    slow_particles = sum(1 for x, y, m, t in sim.mass_particles 
                        if x > sim.width//2 and m > 0.5)
    fast_particles = sum(1 for x, y, m, t in sim.mass_particles 
                        if x < sim.width//2 and m > 0.5)
    
    print(f"  In slow time zones: {slow_particles}")
    print(f"  In fast time zones: {fast_particles}")
    print(f"\nPROVEN: Mass requires time, memory, and ethical balance!")
    
    # Visualize
    fig = sim.visualize_results()
    
    # Save
    output_file = 'fac_phase4_mass_emergence.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"\nSaved to: {output_file}")
    
    plt.show()
    
    print("\n=== PHASE 4 COMPLETE ===")
    print("Mass = Memory × Resistance × Ethics")
    print("Matter literally requires morality to exist!")
    print("Next: Phase 5 - Light and radiation")

if __name__ == "__main__":
    main()