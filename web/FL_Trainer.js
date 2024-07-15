import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Fill-Nodes.FixedSizeAnimatedDisplay",
    async nodeCreated(node) {
        const animatedNodeClasses = [
            "FL_KohyaSSDatasetConfig",
        ];

        if (animatedNodeClasses.includes(node.comfyClass)) {
            addAnimatedDisplay(node);
        }
    }
});

function addAnimatedDisplay(node) {
    const nodeCount = 10;
    const nodes = Array.from({ length: nodeCount }, () => ({
        x: 0.5,
        y: 0.5,
        dx: (Math.random() - 0.5) * 0.02,
        dy: (Math.random() - 0.5) * 0.02,
        color: `hsl(${360 * Math.random()}, 100%, 50%)`
    }));

    const ANIMATION_WIDTH = 50;  // Fixed width for the animation
    const ANIMATION_HEIGHT = -60; // Fixed height for the animation

    node.onDrawBackground = function(ctx) {
        ctx.save();

        // Calculate position to center the animation above the node
        const nodeWidth = this.size[0];
        const xOffset = (nodeWidth - ANIMATION_WIDTH) / 2;
        ctx.translate(xOffset, -ANIMATION_HEIGHT);

        // Clear the canvas with full transparency
        ctx.clearRect(1000, 0, ANIMATION_WIDTH, ANIMATION_HEIGHT);

        // Update and draw nodes
        nodes.forEach(node => {
            node.x += node.dx;
            node.y += node.dy;

            if (node.x < 0 || node.x > 1) {
                node.dx *= -1;
                node.color = `hsl(${360 * Math.random()}, 100%, 50%)`;
            }
            if (node.y < 0 || node.y > 1) {
                node.dy *= -1;
                node.color = `hsl(${360 * Math.random()}, 100%, 50%)`;
            }

            node.x = Math.max(0, Math.min(1, node.x));
            node.y = Math.max(0, Math.min(1, node.y));

            ctx.beginPath();
            ctx.arc(node.x * ANIMATION_WIDTH, node.y * ANIMATION_HEIGHT, 2, 0, Math.PI * 2);
            ctx.fillStyle = node.color;
            ctx.fill();
        });

        // Draw connections
        ctx.lineWidth = 0.1;
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                ctx.beginPath();
                ctx.moveTo(nodes[i].x * ANIMATION_WIDTH, nodes[i].y * ANIMATION_HEIGHT);
                ctx.lineTo(nodes[j].x * ANIMATION_WIDTH, nodes[j].y * ANIMATION_HEIGHT);
                ctx.strokeStyle = 'rgba(255, 255, 255, 1)';
                ctx.stroke();
            }
        }

        ctx.restore();

        this.setDirtyCanvas(true);
        requestAnimationFrame(() => this.setDirtyCanvas(true));
    };

    node.setDirtyCanvas(true);
}