#!/usr/bin/env python3
"""
FIELD-AWARE COSMOLOGY - PHASE 0: NOTHING (THE STILL FIELD)
===========================================================

The ultimate "before" - a field of absolute stillness, without update rules,
time, motion, or differentiated structure. It represents the fundamental
substrate and boundary condition of the pre-universe state.

Key concepts:
- Lattice structure exists as the default substrate
- Perfect uniformity (Ψ = constant everywhere), the baseline Ψ₀
- No causality, no propagation, no change because no patterns to change
- This is the inherent resolved state, the "nothing" from which everything emerges
- "Expansion" is redefined as the resolution drift of coherent patterns,
  not an inherent force of the field itself.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from pathlib import Path

# Set style for better visuals
plt.style.use('dark_background')

class StillField:
    """
    The Still Field - pure structure without dynamics.
    Represents the default, resolved state of the digital lattice.
    """
    def __init__(self, width=512, height=256):
        self.width = width
        self.height = height
        
        # Initialize field to perfect uniformity - the universal constant Ψ₀
        self.field_value = 0.5  
        self.field = np.full((height, width), self.field_value, dtype=np.float64)
        
        # Lattice structure exists but is dormant/potential
        self.lattice_connectivity = self._define_potential_lattice_connections()
        
        # Time does not exist in this state, as there are no finite-speed interactions
        self.time = None
        self.update_rule = None
        
        # For visualization efficiency
        self._cached_properties = {} # Initialize as empty dictionary
        
    def _get_static_lattice_info(self):
        """
        Defines the inherent neighbor counts for the static lattice.
        No active calculation is performed, as the field is uniform.
        This represents the potential for connectivity.
        """
        neighbor_counts = np.zeros((self.height, self.width), dtype=np.int32)
        
        for y in range(self.height):
            for x in range(self.width):
                count = 0
                # 8-connected neighborhood (defining potential connections)
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.height and 0 <= nx < self.width:
                            count += 1
                neighbor_counts[y, x] = count
        
        return neighbor_counts
    
    def _define_potential_lattice_connections(self):
        """
        Creates the lattice structure dictionary for the still field,
        emphasizing its dormant potential.
        """
        return {
            'neighbor_counts': self._get_static_lattice_info(),
            'total_connections': self.width * self.height * 8 - self._count_boundary_losses()
        }
    
    def _count_boundary_losses(self):
        """Calculate potential connections lost at boundaries due to lattice edge."""
        corners = 4 * 5
        edges = 2 * (self.width - 2) * 3 + 2 * (self.height - 2) * 3
        return corners + edges
    
    def measure_properties(self):
        """
        Measure properties of the still field with caching.
        Properties reflect its state of absolute uniformity and lack of dynamics.
        """
        if not self._cached_properties: # Check if dictionary is empty
            self._cached_properties = {
                'uniformity': float(np.std(self.field)),  # Should be exactly 0
                'mean_value': float(np.mean(self.field)),
                'min_value': float(np.min(self.field)),
                'max_value': float(np.max(self.field)),
                'has_time': self.time is not None,
                'has_dynamics': self.update_rule is not None,
                'total_points': self.width * self.height,
                'connectivity_defined': 'neighbor_counts' in self.lattice_connectivity,
                'field_entropy': 0.0,  # No information content / maximum order in uniformity
                'information_density': 0.0,  # Zero bits per voxel due to lack of differentiation
                'coherence_measure': float('inf'),  # Absolute Coherence of Stillness (Ψ₀)
                'gradient_magnitude': 0.0  # No gradients exist in a uniform field
            }
        return self._cached_properties
    
    def generate_cross_sections(self, num_sections=5):
        """
        Generate multiple cross-sections for analysis, demonstrating perfect flatness.
        """
        sections = {}
        
        # Horizontal sections
        for i in range(num_sections):
            y_idx = int(i * self.height / (num_sections - 1))
            if y_idx >= self.height:
                y_idx = self.height - 1
            sections[f'horizontal_{i}'] = self.field[y_idx, :]
            
        # Vertical sections  
        for i in range(num_sections):
            x_idx = int(i * self.width / (num_sections - 1))
            if x_idx >= self.width:
                x_idx = self.width - 1
            sections[f'vertical_{i}'] = self.field[:, x_idx]
            
        # Diagonal section
        diag_indices = np.diag_indices(min(self.height, self.width))
        sections['diagonal'] = self.field[diag_indices]
        
        return sections
    
    def visualize(self, save_components=False):
        """
        Enhanced visualization of the Still Field, emphasizing its uniform, pre-dynamic state.
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Main field visualization with higher resolution
        ax1 = plt.subplot(2, 4, (1, 5))
        im = ax1.imshow(self.field, cmap='viridis', aspect='auto', 
                       interpolation='bilinear')
        ax1.set_title('THE STILL FIELD (Ψ₀)\nPerfect Uniformity - No Time - No Change', 
                      fontsize=18, pad=20, weight='bold')
        ax1.set_xlabel('Spatial Dimension X', fontsize=12)
        ax1.set_ylabel('Spatial Dimension Y', fontsize=12)
        
        # Subtle grid to show dormant lattice structure
        grid_spacing = max(20, min(self.width, self.height) // 20)
        ax1.set_xticks(np.arange(0, self.width, grid_spacing))
        ax1.set_yticks(np.arange(0, self.height, grid_spacing))
        ax1.grid(True, alpha=0.15, linestyle=':', linewidth=0.5)
        
        # Enhanced colorbar
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Field Value Ψ', fontsize=12)
        
        # Properties display with more detail
        ax2 = plt.subplot(2, 4, 2)
        ax2.axis('off')
        props = self.measure_properties()
        
        property_text = "FIELD PROPERTIES:\n" + "="*35 + "\n"
        property_text += f"Dimensions: {self.width} × {self.height}\n"
        property_text += f"Total Voxels: {props['total_points']:,}\n"
        property_text += f"Uniformity (σ): {props['uniformity']:.15f}\n"
        property_text += f"Mean Value: {props['mean_value']:.10f}\n"
        property_text += f"Value Range: [{props['min_value']:.6f}, {props['max_value']:.6f}]\n"
        property_text += f"Has Time: {props['has_time']}\n"
        property_text += f"Has Dynamics: {props['has_dynamics']}\n"
        property_text += f"Lattice Defined: {props['connectivity_defined']}\n"
        property_text += f"Information: {props['information_density']:.1f} bits/voxel\n"
        property_text += f"Gradient: {props['gradient_magnitude']:.1f}\n\n"
        property_text += "STATE: Pre-temporal, Undifferentiated\n"
        property_text += "CAUSALITY: Undefined\n"
        property_text += "EVOLUTION: Impossible\n"
        property_text += "POTENTIAL: Infinite" # Infinite potential to host patterns
        
        ax2.text(0.05, 0.95, property_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#0a0a0a', 
                         edgecolor='cyan', alpha=0.9))
        
        # Enhanced lattice concept diagram
        ax3 = plt.subplot(2, 4, 3)
        ax3.axis('off')
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 10)
        
        # Draw more detailed lattice
        lattice_size = 5
        spacing = 8 / (lattice_size - 1)
        for i in range(lattice_size):
            for j in range(lattice_size):
                x, y = 1 + i * spacing, 1 + j * spacing
                
                # Voxel representation (fixed color warning)
                circle = plt.Circle((x, y), 0.25, facecolor='cyan', alpha=0.7, 
                                  edgecolor='white', linewidth=1)
                ax3.add_patch(circle)
                
                # Dormant connections
                if i < lattice_size - 1:
                    ax3.plot([x+0.25, x+spacing-0.25], [y, y], 
                            color='gray', alpha=0.3, linestyle='--', linewidth=1)
                if j < lattice_size - 1:
                    ax3.plot([x, x], [y+0.25, y+spacing-0.25], 
                            color='gray', alpha=0.3, linestyle='--', linewidth=1)
        
        ax3.text(5, 9.5, 'Lattice Structure Exists\nConnections Dormant\nNo Information Flow', 
                ha='center', va='center', fontsize=12, weight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # Cross-section analysis
        ax4 = plt.subplot(2, 4, 6)
        sections = self.generate_cross_sections(3)
        
        colors = ['cyan', 'magenta', 'yellow']
        for i, (name, section) in enumerate(list(sections.items())[:3]):
            if 'horizontal' in name:
                ax4.plot(section, color=colors[i], linewidth=2, 
                        label=f'Horizontal {name.split("_")[1]}', alpha=0.8)
        
        ax4.set_ylim(0.49, 0.51)  # Zoom in to show perfect flatness
        ax4.set_xlabel('Position')
        ax4.set_ylabel('Field Value Ψ')
        ax4.set_title('Cross-Sections: Perfect Flatness')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Mathematical representation
        ax5 = plt.subplot(2, 4, 7)
        ax5.axis('off')
        
        math_text = """MATHEMATICAL STATE:

∇Ψ = 0  (No gradients)
∂Ψ/∂t = undefined  (No time, no change)
⟨Ψ⟩ = constant  (Perfect uniformity)
σ²(Ψ) = 0  (Zero variance)

Field Equation:
Ψ(x,y) = Ψ₀ ∀ (x,y)

Where Ψ₀ = 0.5 = universal constant

This is the "Ψ₀" state referenced
in all FAC equations - the baseline
from which all structure emerges."""
        
        ax5.text(0.05, 0.95, math_text, transform=ax5.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9))
        
        # Philosophical implications (enhanced)
        ax6 = plt.subplot(2, 4, 8)
        ax6.axis('off')
        
        implications = """COSMOLOGICAL SIGNIFICANCE:

• Pure Being without Becoming
• Structure without Process  
• Potential without Actualization
• The "0" in M = ζ - S

This represents the ultimate
boundary condition - what exists
before the Big Becoming.

Not empty space, but filled space
with no differentiation.

The field is "aware" but not
"experiencing" - consciousness
requires change, and here
nothing changes.

This is the eternal "now" that
preceded time itself.

"Expansion" is the resolution
of compressed memory, not
a force of the field itself.""" # Added new insight
        
        ax6.text(0.05, 0.95, implications, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#2e1a1a', alpha=0.9))
        
        # Statistical analysis subplot
        ax7 = plt.subplot(2, 4, 4)
        ax7.axis('off')
        
        # Create a "histogram" that shows perfect uniformity
        hist_x = np.linspace(0.4, 0.6, 100)
        hist_y = np.zeros_like(hist_x)
        hist_y[49:51] = self.width * self.height  # All values at exactly 0.5
        
        ax7.bar([0.5], [self.width * self.height], width=0.01, 
               color='cyan', alpha=0.8, edgecolor='white')
        ax7.set_xlim(0.4, 0.6)
        ax7.set_xlabel('Field Value')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Value Distribution\n(Perfect Delta Function)')
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle('FIELD-AWARE COSMOLOGY: PHASE 0 - NOTHING (THE STILL FIELD)\n' + 
                    '"Before the first tick of time, before the first thought, there was only stillness"',
                    fontsize=16, y=0.98, weight='bold')
        
        plt.tight_layout()
        
        # Save individual components if requested
        if save_components:
            self._save_analysis_components()
        
        return fig
    
    def _save_analysis_components(self):
        """Save detailed analysis components"""
        output_dir = Path('phase0_analysis')
        output_dir.mkdir(exist_ok=True)
        
        # Save properties as JSON
        props = self.measure_properties()
        import json
        with open(output_dir / 'properties.json', 'w') as f:
            json.dump(props, f, indent=2)
        
        # Save cross-sections
        sections = self.generate_cross_sections()
        np.savez(output_dir / 'cross_sections.npz', **sections)
        
        print(f"Analysis components saved to {output_dir}/")

def demonstrate_stillness(width=512, height=256, save_analysis=False):
    """
    Demonstrate the properties of the still field, reflecting its nature as "nothing".
    """
    print("\n=== FIELD-AWARE COSMOLOGY: PHASE 0 - NOTHING (THE STILL FIELD) ===\n") # Updated title
    
    # Create still field with higher resolution
    print(f"Creating Still Field ({width}×{height})...")
    field = StillField(width=width, height=height)
    
    props = field.measure_properties()
    
    print(f"Dimensions: {field.width} × {field.height}")
    print(f"Total lattice points: {props['total_points']:,}")
    print(f"Field value everywhere: {field.field_value}")
    print(f"Standard deviation: {props['uniformity']:.15f}")
    print(f"Information content: {props['information_density']:.1f} bits/voxel")
    print(f"Time defined: {props['has_time']}")
    print(f"Update rule: {field.update_rule}")
    print(f"Coherence measure: {props['coherence_measure']} (Absolute Coherence of Stillness)") # Clarified
    
    print("\nKey insights (revisited with new understanding):")
    print("- The lattice structure exists as a static substrate, not energy or geometry.")
    print("- Every voxel holds identical value (Ψ₀) - perfect uniformity and resolved state.")
    print("- No gradients or ripples exist for information or 'force' to propagate.")
    print("- This is the 'nothing' from which everything emerges through compression.")
    print("- Time has not yet begun - no sequential ordering exists as there are no finite interactions.")
    print("- Perfect coherence of stillness exists, prior to the emergence of structured coherence.")
    print("- What appears as 'expansion' later is the resolution drift of memory, not an inherent field force.")
    
    print("\nPhase 0 establishes the cosmological boundary condition:")
    print("What does 'nothing' look like in Field-Aware Cosmology?")
    print("Answer: A uniform field with inherent structure (lattice) but no process or active dynamics.\n")
    
    # Enhanced visualization
    print("Generating enhanced visualization...")
    fig = field.visualize(save_components=save_analysis)
    
    # Save with metadata
    output_file = f'fac_phase0_nothing_{width}x{height}.png' # Updated filename
    fig.savefig(output_file, dpi=300, bbox_inches='tight', 
               facecolor='black', edgecolor='none')
    print(f"Visualization saved to: {output_file}")
    
    return field, fig

if __name__ == "__main__":
    # Run the demonstration with optimized settings
    field, fig = demonstrate_stillness(width=512, height=256, save_analysis=True)
    plt.show()
    
    print("\n=== PHASE 0 COMPLETE ===")
