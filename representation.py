import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def visualize_lograph_architecture(save_path="lograph_architecture.png"):
    """
    Visualize the Lograph model architecture
    """
    # CREATE FIGURE: width=16 inches, height=24 inches, resolution=150 DPI
    fig, ax = plt.subplots(figsize=(16, 24), dpi=150)
    
    # SET COORDINATE SYSTEM: x from 0-10, y from 0-26
    # x: horizontal position (0=left, 10=right)
    # y: vertical position (0=bottom, 26=top)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 26)
    ax.axis('off')  # Hide axes
    
    # DEFINE COLORS for different layer types
    colors = {
        'input': '#E3F2FD',
        'embedding': '#BBDEFB',
        'attention': '#FFE082',
        'aggregation': '#FFB74D',
        'classifier': '#FF8A65',
        'output': '#81C784'
    }
    
    # HELPER FUNCTION: Draw a box
    # x, y: center position of the box
    # width, height: size of the box
    # Returns: (bottom_y, top_y) so we know where edges are
    def draw_box(x, y, width, height, text, color, fontsize=11):
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,  # Position: center minus half size
            boxstyle="round,pad=0.1", 
            edgecolor='black', facecolor=color,
            linewidth=2.5, alpha=0.9
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold', wrap=True)
        return y - height/2, y + height/2  # Return bottom and top edges
    
    # HELPER FUNCTION: Draw an arrow
    # (x1, y1): start point coordinates
    # (x2, y2): end point coordinates
    def draw_arrow(x1, y1, x2, y2, style='->', color='black', linewidth=2.5):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, color=color, linewidth=linewidth,
            mutation_scale=25, alpha=0.8
        )
        ax.add_patch(arrow)
    
    # START BUILDING THE DIAGRAM FROM TOP
    # y coordinate decreases as we go down
    y = 24.5  # Start near the top
    
    # ========== 1. INPUT LAYER ==========
    # Draw box at x=5 (center), y=24.5, width=7, height=1.2
    bottom_input, top_input = draw_box(5, y, 7, 1.2, 
                                        'Input: Log-Entity Graph Snapshot', 
                                        colors['input'], 13)
    # bottom_input is now at y ≈ 23.9
    # top_input is now at y ≈ 25.1
    
    # FIX: MORE SPACE between Input and Embedding
    y_embed = bottom_input - 2.0  # INCREASE this number for MORE space (was 1.3, now 2.0)
    # y_embed is now at ≈ 21.9
    
    # Draw arrow from bottom of Input to top of Embedding
    # Start: (5, 23.9) → End: (5, 21.9 + 0.95) = (5, 22.85)
    draw_arrow(5, bottom_input - 0.05, 5, y_embed + 0.95)
    
    # ========== 2. EMBEDDING LAYER ==========
    # Draw box at x=5, y=21.9, width=8, height=1.8
    bottom_embed, top_embed = draw_box(5, y_embed, 8, 1.8, 
             'LographEmbeddingLayer\n(GloVe word embeddings + TF-IDF aggregation)', 
             colors['embedding'], 12)
    # bottom_embed is now at y ≈ 21.0
    # top_embed is now at y ≈ 22.8
    
    # FIX: MORE SPACE before 5 small boxes
    y_small_boxes = bottom_embed - 2.5  # INCREASE for MORE space (was 2.0, now 2.5)
    # y_small_boxes is now at ≈ 18.5
    
    # ========== 3. FIVE OUTPUT BOXES FROM EMBEDDING ==========
    # Draw 5 small boxes horizontally spread out
    small_box_positions = [
        (1.5, 'Log\nRepresentations'),    # x=1.5 (left side)
        (3, 'Log\nAdjacency'),            # x=3
        (5, 'Entity\nIndices'),           # x=5 (center)
        (7, 'Entity\nRepresentations'),   # x=7
        (8.5, 'Entity\nAdjacency')        # x=8.5 (right side)
    ]
    
    small_box_bottoms = {}  # Store bottom positions for later
    for x_pos, label in small_box_positions:
        # Draw arrow from embedding (x=5) to each small box (x=x_pos)
        # Arrow fans out from center to different x positions
        draw_arrow(5, bottom_embed - 0.05, x_pos, y_small_boxes + 0.55)
        
        # Draw small box at x=x_pos, y=18.5, width=1.3, height=1
        bottom_small, top_small = draw_box(x_pos, y_small_boxes, 1.3, 1, 
                                           label, colors['embedding'], 9)
        small_box_bottoms[x_pos] = bottom_small  # Save bottom position
    # small_box_bottoms[1.5] is now at ≈ 18.0
    
    # FIX: MORE SPACE before meta-path layers
    y_meta = y_small_boxes - 2.5  # INCREASE for MORE space (was 1.5, now 2.5)
    # y_meta is now at ≈ 16.0 (this is the CENTER of meta-path boxes)
    
    # ========== 4. META-PATH LAYERS (Left and Right) ==========
    
    # LEFT BRANCH: LogEntityLogLayer
    # FIX: Arrow from bottom of "Log Representations" (x=1.5) to top of left meta-path box
    draw_arrow(1.5, small_box_bottoms[1.5] - 0.05, 2, y_meta + 1.3)
    # Arrow goes from (1.5, 18.0) to (2, 17.3)
    
    # Draw left meta-path box at x=2, y=16.0, width=3.2, height=2.5
    bottom_lel, top_lel = draw_box(2, y_meta, 3.2, 2.5, 
             'LogEntityLogLayer\n(meta-path 1)\n\nGRU + Attention\nor\nTransformer', 
             colors['attention'], 11)
    # top_lel is at ≈ 17.25
    # bottom_lel is at ≈ 14.75
    
    # Output box for left branch
    y_lel_out = bottom_lel - 2.0  # INCREASE for MORE space
    draw_arrow(2, bottom_lel - 0.05, 2, y_lel_out + 0.55)
    bottom_lel_out, top_lel_out = draw_box(2, y_lel_out, 2.5, 1, 
                                            'att_log_reprs', colors['attention'], 10)
    
    # RIGHT BRANCH: EntityLogEntityLayer
    # FIX: Arrow from bottom of "Entity Adjacency" (x=8.5) to top of right meta-path box
    draw_arrow(8.5, small_box_bottoms[8.5] - 0.05, 8, y_meta + 1.3)
    # Arrow goes from (8.5, 18.0) to (8, 17.3)
    
    # Draw right meta-path box at x=8, y=16.0, width=3.2, height=2.5
    bottom_ele, top_ele = draw_box(8, y_meta, 3.2, 2.5, 
             'EntityLogEntityLayer\n(meta-path 2)\n\nSelf-Attention + GAT', 
             colors['attention'], 11)
    
    # Output box for right branch
    y_ele_out = bottom_ele - 2.0  # INCREASE for MORE space
    draw_arrow(8, bottom_ele - 0.05, 8, y_ele_out + 0.55)
    bottom_ele_out, top_ele_out = draw_box(8, y_ele_out, 2.5, 1, 
                                            'att_entity_reprs', colors['attention'], 10)
    
    # ========== 5. SEMANTIC AGGREGATION LAYER ==========
    # Position below the lower of the two att output boxes
    y_agg = min(bottom_lel_out, bottom_ele_out) - 2.0  # INCREASE for MORE space
    
    # Arrows from both att outputs converge to aggregation box
    draw_arrow(2, bottom_lel_out - 0.05, 5, y_agg + 0.8)
    draw_arrow(8, bottom_ele_out - 0.05, 5, y_agg + 0.8)
    
    bottom_agg, top_agg = draw_box(5, y_agg, 6, 1.5, 
             'SemanticAggregationLayer\n(Weighted sum of meta-path representations)', 
             colors['aggregation'], 11)
    
    # Output of aggregation
    y_agg_out = bottom_agg - 1.8  # INCREASE for MORE space
    draw_arrow(5, bottom_agg - 0.05, 5, y_agg_out + 0.55)
    bottom_agg_out, top_agg_out = draw_box(5, y_agg_out, 2.5, 1, 
                                            'agg_reprs', colors['aggregation'], 11)
    
    # ========== 6. CLASSIFIER ==========
    y_class = bottom_agg_out - 1.8  # INCREASE for MORE space
    draw_arrow(5, bottom_agg_out - 0.05, 5, y_class + 0.65)
    bottom_class, top_class = draw_box(5, y_class, 3.5, 1.2, 
                                        'Classifier\n(Linear Layer)', 
                                        colors['classifier'], 12)
    
    # ========== 7. SOFTMAX ==========
    y_soft = bottom_class - 1.8  # INCREASE for MORE space
    draw_arrow(5, bottom_class - 0.05, 5, y_soft + 0.55)
    bottom_soft, top_soft = draw_box(5, y_soft, 2.5, 1, 
                                      'Softmax', colors['classifier'], 11)
    
    # ========== 8. OUTPUT ==========
    y_out = bottom_soft - 1.8  # INCREASE for MORE space
    draw_arrow(5, bottom_soft - 0.05, 5, y_out + 0.65)
    draw_box(5, y_out, 5, 1.2, 'Output: [P(normal), P(anomaly)]', 
             colors['output'], 12)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['input'], edgecolor='black', label='Input Layer', linewidth=2),
        mpatches.Patch(facecolor=colors['embedding'], edgecolor='black', label='Embedding Layer', linewidth=2),
        mpatches.Patch(facecolor=colors['attention'], edgecolor='black', label='Attention Layers', linewidth=2),
        mpatches.Patch(facecolor=colors['aggregation'], edgecolor='black', label='Aggregation Layer', linewidth=2),
        mpatches.Patch(facecolor=colors['classifier'], edgecolor='black', label='Classifier', linewidth=2),
        mpatches.Patch(facecolor=colors['output'], edgecolor='black', label='Output', linewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=11, framealpha=0.95)
    
    plt.title('Lograph Model Architecture', fontsize=20, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Architecture diagram saved to {save_path}")