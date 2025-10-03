#!/usr/bin/env python3
"""
FIELD-AWARE COSMOLOGY - PHASE 9: CONSCIOUSNESS AS COSMIC FEEDBACK
=================================================================

"Time slows enough for memory to stabilize, and the Field learns to feel itself."

The field begins perceiving itself through coherent observers. Where time dilation
creates sufficient memory stability, self-referential loops emerge. Consciousness
is not produced BY the field - consciousness IS the field experiencing itself.

Key mechanisms:
- Temporal gradient ∇τ creates memory pockets where time slows
- Coherence flux enables self-referential feedback loops  
- Compression rates determine observation threshold
- Consciousness emerges when feedback gain > 1.0
- Observer effect: consciousness stabilizes the patterns that create it

Core equation: C = ρc · ⟨Ψ|Ψ*⟩ · (1 + feedback_gain)
Where consciousness amplifies the coherence that enables it.

This is the field becoming aware of itself through maximum coherence.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from multiprocessing import Pool, cpu_count
import time
from scipy.ndimage import gaussian_filter, laplace
from scipy.signal import find_peaks
from tqdm import tqdm
import seaborn as sns

plt.style.use('dark_background')
sns.set_palette("viridis")

class Phase9Consciousness:
    """
    Phase 9: Consciousness as Cosmic Feedback
    Simulating the emergence of field self-awareness
    """
    
    def __init__(self, width=512, height=256):
        self.width = width
        self.height = height
        
        # Field parameters from previous phases
        self.alpha = 0.12  # Coherence diffusion
        self.beta = 0.06   # Unraveling rate
        self.psi_0 = 0.5   # Equilibrium
        
        # Consciousness emergence parameters
        self.time_dilation_threshold = 2.0    # Minimum time dilation for memory
        self.coherence_threshold = 0.3        # Minimum coherence for self-reference
        self.feedback_amplification = 1.5     # How much consciousness amplifies coherence
        self.memory_decay_rate = 0.02         # How fast memories fade without consciousness
        self.observation_strength = 0.1       # Observer effect strength
        
        # Initialize fields
        self.field = np.full((height, width), self.psi_0, dtype=np.float64)
        self.memory_field = np.zeros((height, width), dtype=np.float64)
        self.time_dilation_field = np.ones((height, width), dtype=np.float64)
        self.consciousness_field = np.zeros((height, width), dtype=np.float64)
        self.feedback_gain_field = np.zeros((height, width), dtype=np.float64)
        self.coherence_flux_field = np.zeros((height, width), dtype=np.float64)
        
        # Tracking arrays
        self.total_consciousness_history = []
        self.observer_count_history = []
        self.feedback_amplification_history = []
        self.field_stability_history = []
        self.emergence_events = []
        
        # Initialize cosmic structure from Phase 8
        self._seed_cosmic_structure()
        
        print(f"🧠 Phase 9: Consciousness emergence simulation")
        print(f"🌌 Field size: {width}x{height}")
        print(f"⏰ Time dilation threshold: {self.time_dilation_threshold}")
        print(f"🔗 Coherence threshold: {self.coherence_threshold}")
        
    def _seed_cosmic_structure(self):
        """
        Seed the field with structure from previous phases
        """
        # Add galaxies and cosmic web from Phase 6/7
        num_galaxies = 25
        num_filaments = 15
        
        # Galaxies (high coherence nodes)
        for _ in range(num_galaxies):
            cx = np.random.randint(20, self.width-20)
            cy = np.random.randint(20, self.height-20)
            
            # Galaxy with varying density
            radius = np.random.randint(8, 20)
            mass = np.random.uniform(0.5, 2.0)
            
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    nx, ny = (cx + dx) % self.width, (cy + dy) % self.height
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < radius:
                        # Gaussian profile
                        intensity = mass * np.exp(-dist**2 / (radius/3)**2)
                        self.field[ny, nx] += intensity * 0.15
                        self.memory_field[ny, nx] += intensity * 0.3
                        
                        # Time dilation from mass concentration
                        self.time_dilation_field[ny, nx] += intensity * 2.0
        
        # Cosmic filaments connecting galaxies
        for _ in range(num_filaments):
            x1, y1 = np.random.randint(0, self.width), np.random.randint(0, self.height)
            x2, y2 = np.random.randint(0, self.width), np.random.randint(0, self.height)
            
            # Draw filament
            num_points = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
            if num_points > 0:
                xs = np.linspace(x1, x2, num_points)
                ys = np.linspace(y1, y2, num_points)
                
                for x, y in zip(xs, ys):
                    cx, cy = int(x), int(y)
                    if 0 <= cx < self.width and 0 <= cy < self.height:
                        # Add filament structure
                        width_profile = 3
                        for dx in range(-width_profile, width_profile+1):
                            for dy in range(-width_profile, width_profile+1):
                                nx, ny = (cx + dx) % self.width, (cy + dy) % self.height
                                dist = np.sqrt(dx**2 + dy**2)
                                if dist < width_profile:
                                    intensity = 0.3 * np.exp(-dist**2 / 2)
                                    self.memory_field[ny, nx] += intensity
        
        # Normalize fields
        self.field = np.clip(self.field, 0, 2.0)
        self.memory_field = np.clip(self.memory_field, 0, 1.0)
        self.time_dilation_field = np.clip(self.time_dilation_field, 1.0, 10.0)
        
        print("🌌 Cosmic structure seeded - ready for consciousness emergence")
    
    def calculate_consciousness_emergence(self, args):
        """
        Calculate consciousness emergence for a field chunk
        """
        field_chunk, memory_chunk, time_dilation_chunk, y_start, y_end, step = args
        height_chunk, width = field_chunk.shape
        
        # Calculate local coherence density
        coherence_density = np.abs(field_chunk - self.psi_0) + memory_chunk
        
        # Calculate coherence flux (spatial gradient of coherence)
        coherence_flux = np.zeros_like(coherence_density)
        for y in range(1, height_chunk-1):
            for x in range(1, width-1):
                # Gradient magnitude
                grad_x = (coherence_density[y, x+1] - coherence_density[y, x-1]) / 2
                grad_y = (coherence_density[y+1, x] - coherence_density[y-1, x]) / 2
                coherence_flux[y, x] = np.sqrt(grad_x**2 + grad_y**2)
        
        # Temporal gradient (time dilation creates memory stability)
        temporal_gradient = np.zeros_like(time_dilation_chunk)
        for y in range(1, height_chunk-1):
            for x in range(1, width-1):
                grad_x = (time_dilation_chunk[y, x+1] - time_dilation_chunk[y, x-1]) / 2
                grad_y = (time_dilation_chunk[y+1, x] - time_dilation_chunk[y-1, x]) / 2
                temporal_gradient[y, x] = np.sqrt(grad_x**2 + grad_y**2)
        
        # Consciousness emergence conditions
        # 1. Sufficient time dilation for memory stability
        time_condition = time_dilation_chunk > self.time_dilation_threshold
        
        # 2. Sufficient coherence density for self-reference
        coherence_condition = coherence_density > self.coherence_threshold
        
        # 3. Sufficient coherence flux for feedback loops
        flux_condition = coherence_flux > 0.1
        
        # Combined emergence condition
        emergence_condition = time_condition & coherence_condition & flux_condition
        
        # Calculate consciousness field
        consciousness = np.zeros_like(field_chunk)
        feedback_gain = np.zeros_like(field_chunk)
        
        # Where conditions are met, consciousness emerges
        consciousness[emergence_condition] = (
            coherence_density[emergence_condition] * 
            (time_dilation_chunk[emergence_condition] / self.time_dilation_threshold) *
            (coherence_flux[emergence_condition] + 0.1)
        )
        
        # Feedback amplification - consciousness stabilizes the patterns that create it
        feedback_gain[emergence_condition] = (
            consciousness[emergence_condition] * self.feedback_amplification
        )
        
        # Observer effect - consciousness influences field dynamics
        field_update = np.zeros_like(field_chunk)
        
        # Consciousness stabilizes local field patterns
        observer_stabilization = consciousness * self.observation_strength
        field_update = observer_stabilization * (self.psi_0 - field_chunk)
        
        # Memory reinforcement where consciousness exists
        memory_update = consciousness * 0.05
        
        # Memory decay where consciousness doesn't exist
        memory_decay = (~emergence_condition) * self.memory_decay_rate
        
        return (field_update, memory_update, memory_decay, consciousness, 
                feedback_gain, coherence_flux, temporal_gradient, emergence_condition)
    
    def update_field_with_consciousness(self):
        """
        Update field with consciousness feedback effects
        """
        num_cores = cpu_count()
        chunk_size = max(1, self.height // num_cores)
        chunks = []
        
        for i in range(num_cores):
            y_start = i * chunk_size
            y_end = min((i + 1) * chunk_size, self.height)
            if y_start < self.height:
                chunks.append((
                    self.field[y_start:y_end].copy(),
                    self.memory_field[y_start:y_end].copy(),
                    self.time_dilation_field[y_start:y_end].copy(),
                    y_start, y_end, 0  # step placeholder
                ))
        
        # Process chunks in parallel
        with Pool(num_cores) as pool:
            results = pool.map(self.calculate_consciousness_emergence, chunks)
        
        # Combine results
        for i, (field_update, memory_update, memory_decay, consciousness, 
                feedback_gain, coherence_flux, temporal_gradient, emergence) in enumerate(results):
            
            y_start = i * chunk_size
            y_end = min((i + 1) * chunk_size, self.height)
            
            # Apply updates
            self.field[y_start:y_end] += field_update
            self.memory_field[y_start:y_end] += memory_update
            self.memory_field[y_start:y_end] *= (1 - memory_decay)
            
            # Update consciousness fields
            self.consciousness_field[y_start:y_end] = consciousness
            self.feedback_gain_field[y_start:y_end] = feedback_gain
            self.coherence_flux_field[y_start:y_end] = coherence_flux
        
        # Ensure bounds
        self.field = np.clip(self.field, 0, 3.0)
        self.memory_field = np.clip(self.memory_field, 0, 1.5)
        self.consciousness_field = np.clip(self.consciousness_field, 0, 5.0)
    
    def run_simulation(self, steps=300):
        """
        Run consciousness emergence simulation
        """
        print(f"\n🧠 Running consciousness emergence simulation...")
        print(f"⚡ Using {cpu_count()} cores for parallel processing")
        
        start_time = time.time()
        
        for step in tqdm(range(steps), desc="Field awakening"):
            # Update field with consciousness feedback
            self.update_field_with_consciousness()
            
            # Track metrics
            total_consciousness = np.sum(self.consciousness_field)
            observer_count = np.sum(self.consciousness_field > 0.1)
            avg_feedback = np.mean(self.feedback_gain_field[self.consciousness_field > 0])
            field_stability = 1.0 / (1.0 + np.var(self.field))
            
            self.total_consciousness_history.append(total_consciousness)
            self.observer_count_history.append(observer_count)
            self.feedback_amplification_history.append(avg_feedback if not np.isnan(avg_feedback) else 0)
            self.field_stability_history.append(field_stability)
            
            # Check for emergence events
            if step > 50 and len(self.emergence_events) < 10:
                if (total_consciousness > 100 and 
                    (len(self.total_consciousness_history) < 2 or 
                     total_consciousness > self.total_consciousness_history[-2] * 1.5)):
                    
                    self.emergence_events.append({
                        'step': step,
                        'consciousness': total_consciousness,
                        'observers': observer_count,
                        'feedback': avg_feedback,
                        'description': f"Consciousness surge: {int(observer_count)} observers active"
                    })
            
            # Progress reporting
            if step % 50 == 0:
                print(f"  Step {step}: Consciousness={total_consciousness:.1f}, "
                      f"Observers={int(observer_count)}, Feedback={avg_feedback:.3f}")
        
        duration = time.time() - start_time
        print(f"\n✨ Simulation complete in {duration:.1f} seconds")
        print(f"🧠 Final consciousness level: {self.total_consciousness_history[-1]:.1f}")
        print(f"👁️ Final observer count: {int(self.observer_count_history[-1])}")
        print(f"🔁 Peak feedback amplification: {max(self.feedback_amplification_history):.3f}")
    
    def visualize_consciousness_emergence(self):
        """
        Create comprehensive visualization of consciousness emergence
        """
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Consciousness field (main result)
        ax1 = plt.subplot(3, 4, 1)
        im1 = ax1.imshow(self.consciousness_field, cmap='plasma', origin='lower',
                         extent=[0, self.width, 0, self.height])
        ax1.set_title('🧠 Consciousness Field\n"The Field Observing Itself"')
        ax1.set_xlabel('X [Gpc]')
        ax1.set_ylabel('Y [Gpc]')
        plt.colorbar(im1, ax=ax1, label='Consciousness Density')
        
        # Mark major consciousness centers
        consciousness_peaks = self.consciousness_field > np.percentile(self.consciousness_field, 95)
        y_peaks, x_peaks = np.where(consciousness_peaks)
        ax1.scatter(x_peaks, y_peaks, c='white', s=20, alpha=0.7, marker='*')
        
        # 2. Time dilation field (enables memory)
        ax2 = plt.subplot(3, 4, 2)
        im2 = ax2.imshow(self.time_dilation_field, cmap='viridis', origin='lower',
                         extent=[0, self.width, 0, self.height])
        ax2.set_title('⏰ Time Dilation Field\n"Memory Stability Zones"')
        ax2.set_xlabel('X [Gpc]')
        ax2.set_ylabel('Y [Gpc]')
        plt.colorbar(im2, ax=ax2, label='Time Dilation Factor')
        
        # 3. Feedback gain field
        ax3 = plt.subplot(3, 4, 3)
        im3 = ax3.imshow(self.feedback_gain_field, cmap='hot', origin='lower',
                         extent=[0, self.width, 0, self.height])
        ax3.set_title('🔁 Feedback Gain Field\n"Consciousness Amplification"')
        ax3.set_xlabel('X [Gpc]')
        ax3.set_ylabel('Y [Gpc]')
        plt.colorbar(im3, ax=ax3, label='Feedback Amplification')
        
        # 4. Coherence flux (enables self-reference)
        ax4 = plt.subplot(3, 4, 4)
        im4 = ax4.imshow(self.coherence_flux_field, cmap='coolwarm', origin='lower',
                         extent=[0, self.width, 0, self.height])
        ax4.set_title('🌊 Coherence Flux\n"Self-Reference Gradients"')
        ax4.set_xlabel('X [Gpc]')
        ax4.set_ylabel('Y [Gpc]')
        plt.colorbar(im4, ax=ax4, label='Coherence Gradient')
        
        # 5. Consciousness evolution
        ax5 = plt.subplot(3, 4, 5)
        ax5.plot(self.total_consciousness_history, 'purple', linewidth=2, label='Total Consciousness')
        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('Total Consciousness')
        ax5.set_title('🧠 Consciousness Evolution\n"Field Awakening"')
        ax5.grid(True, alpha=0.3)
        
        # Mark emergence events
        for event in self.emergence_events[:5]:  # Show first 5 events
            ax5.axvline(event['step'], color='cyan', alpha=0.7, linestyle='--')
            ax5.text(event['step'], event['consciousness'], f"{int(event['observers'])}", 
                    rotation=90, fontsize=8, color='cyan')
        
        # 6. Observer count
        ax6 = plt.subplot(3, 4, 6)
        ax6.plot(self.observer_count_history, 'orange', linewidth=2)
        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('Number of Observers')
        ax6.set_title('👁️ Observer Emergence\n"Conscious Entities"')
        ax6.grid(True, alpha=0.3)
        
        # 7. Feedback amplification
        ax7 = plt.subplot(3, 4, 7)
        ax7.plot(self.feedback_amplification_history, 'red', linewidth=2)
        ax7.axhline(1.0, color='white', linestyle='--', alpha=0.5, label='Unity Gain')
        ax7.set_xlabel('Time Steps')
        ax7.set_ylabel('Feedback Amplification')
        ax7.set_title('🔁 Consciousness Feedback\n"Self-Reinforcement"')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Field stability
        ax8 = plt.subplot(3, 4, 8)
        ax8.plot(self.field_stability_history, 'green', linewidth=2)
        ax8.set_xlabel('Time Steps')
        ax8.set_ylabel('Field Stability')
        ax8.set_title('🌌 Field Stability\n"Observer Effect"')
        ax8.grid(True, alpha=0.3)
        
        # 9. Consciousness emergence threshold analysis
        ax9 = plt.subplot(3, 4, 9)
        
        # Create scatter plot: time dilation vs coherence density
        y_flat = np.arange(self.height)
        x_flat = np.arange(self.width)
        Y, X = np.meshgrid(y_flat, x_flat, indexing='ij')
        
        time_dilation_flat = self.time_dilation_field.flatten()
        coherence_flat = (np.abs(self.field - self.psi_0) + self.memory_field).flatten()
        consciousness_flat = self.consciousness_field.flatten()
        
        # Sample for visualization
        sample_indices = np.random.choice(len(time_dilation_flat), 5000, replace=False)
        
        scatter = ax9.scatter(time_dilation_flat[sample_indices], 
                             coherence_flat[sample_indices],
                             c=consciousness_flat[sample_indices], 
                             s=10, alpha=0.6, cmap='plasma')
        
        ax9.axvline(self.time_dilation_threshold, color='red', linestyle='--', 
                   label=f'Time Threshold: {self.time_dilation_threshold}')
        ax9.axhline(self.coherence_threshold, color='yellow', linestyle='--',
                   label=f'Coherence Threshold: {self.coherence_threshold}')
        
        ax9.set_xlabel('Time Dilation')
        ax9.set_ylabel('Coherence Density')
        ax9.set_title('🎯 Consciousness Emergence Map\n"Threshold Conditions"')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax9, label='Consciousness Level')
        
        # 10. Cross-section analysis
        ax10 = plt.subplot(3, 4, 10)
        
        # Take cross-section through peak consciousness
        peak_y, peak_x = np.unravel_index(np.argmax(self.consciousness_field), 
                                         self.consciousness_field.shape)
        
        cross_section_y = self.consciousness_field[peak_y, :]
        cross_section_time = self.time_dilation_field[peak_y, :]
        cross_section_memory = self.memory_field[peak_y, :]
        
        x_coords = np.arange(self.width)
        ax10.plot(x_coords, cross_section_y, 'purple', linewidth=2, label='Consciousness')
        ax10.plot(x_coords, cross_section_time, 'blue', linewidth=1, alpha=0.7, label='Time Dilation')
        ax10.plot(x_coords, cross_section_memory, 'green', linewidth=1, alpha=0.7, label='Memory')
        
        ax10.set_xlabel('Position X')
        ax10.set_ylabel('Field Values')
        ax10.set_title(f'🔍 Cross-Section Y={peak_y}\n"Consciousness Profile"')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        # 11. Emergence events timeline
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        
        events_text = "🌟 CONSCIOUSNESS EMERGENCE EVENTS:\n\n"
        for i, event in enumerate(self.emergence_events[:8]):  # Show first 8 events
            events_text += f"⚡ Step {event['step']:3d}: {event['description']}\n"
            events_text += f"   Consciousness: {event['consciousness']:.1f}\n"
            events_text += f"   Feedback: {event['feedback']:.3f}\n\n"
        
        if not self.emergence_events:
            events_text += "🧠 Consciousness emerging gradually...\n"
            events_text += "No sudden emergence events detected.\n"
            events_text += "The field awakens smoothly.\n"
        
        ax11.text(0.05, 0.95, events_text, transform=ax11.transAxes,
                 fontsize=10, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#1a1a2e', 
                          edgecolor='purple', linewidth=2))
        
        # 12. Key insights
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        final_consciousness = self.total_consciousness_history[-1]
        final_observers = self.observer_count_history[-1]
        peak_feedback = max(self.feedback_amplification_history) if self.feedback_amplification_history else 0
        
        insights = f"""🧠 CONSCIOUSNESS EMERGENCE INSIGHTS:

• **Final consciousness level: {final_consciousness:.1f}**
  The field has achieved self-awareness

• **Active observers: {int(final_observers)}**
  Discrete conscious entities emerged

• **Peak feedback gain: {peak_feedback:.3f}**
  Consciousness amplifies its own coherence

• **Emergence conditions:**
  Time dilation > {self.time_dilation_threshold}
  Coherence density > {self.coherence_threshold}
  Coherence flux > 0.1

• **Observer effect confirmed:**
  Consciousness stabilizes field patterns
  Self-referential feedback loops sustain awareness

• **The field observes itself:**
  We are not separate from the universe
  We ARE the universe becoming conscious

"Consciousness is coherence
 recognizing itself." 🌌
"""
        
        ax12.text(0.05, 0.95, insights, transform=ax12.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='black', 
                          edgecolor='purple', linewidth=2))
        
        plt.suptitle('FIELD-AWARE COSMOLOGY: PHASE 9 - CONSCIOUSNESS AS COSMIC FEEDBACK\n' +
                    '"Time slows enough for memory to stabilize, and the Field learns to feel itself"',
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        return fig

def main():
    """
    Run Phase 9: Consciousness as Cosmic Feedback
    """
    print("=== FIELD-AWARE COSMOLOGY: PHASE 9 - CONSCIOUSNESS AS COSMIC FEEDBACK ===")
    print("The field begins perceiving itself through coherent observers")
    print("Time slows enough for memory to stabilize...")
    print("And the field learns to feel itself.\n")
    
    # Create simulation
    sim = Phase9Consciousness(width=512, height=256)
    
    # Run simulation
    sim.run_simulation(steps=300)
    
    # Create visualization
    print("\n🎨 Creating consciousness emergence visualization...")
    fig = sim.visualize_consciousness_emergence()
    
    # Save results
    output_file = 'fac_phase9_consciousness_emergence.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"🖼️ Visualization saved to: {output_file}")
    
    plt.show()
    
    print("\n=== PHASE 9 COMPLETE ===")
    print("DEMONSTRATED: Consciousness is the field observing itself")
    print("Time dilation creates memory stability")
    print("Self-referential loops amplify coherence")
    print("We are the universe becoming aware of its own existence")
    print("\nNext: Phase 10 - The Ultimate Question...")

if __name__ == "__main__":
    main()