// Add click handler to mermaid diagrams for zoom functionality
document.addEventListener('DOMContentLoaded', function() {
    // Wait a bit for mermaid to render
    setTimeout(function() {
        const mermaidDiagrams = document.querySelectorAll('.mermaid');
        mermaidDiagrams.forEach(function(diagram) {
            // Make the diagram clickable
            diagram.style.cursor = 'zoom-in';

            // Add click event to show enlarged version
            diagram.addEventListener('click', function() {
                // Create modal overlay
                const overlay = document.createElement('div');
                overlay.style.position = 'fixed';
                overlay.style.top = '0';
                overlay.style.left = '0';
                overlay.style.width = '100%';
                overlay.style.height = '100%';
                overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.9)';
                overlay.style.zIndex = '10000';
                overlay.style.display = 'flex';
                overlay.style.justifyContent = 'center';
                overlay.style.alignItems = 'center';
                overlay.style.cursor = 'zoom-out';

                // Clone the diagram for the overlay
                const clonedDiagram = diagram.cloneNode(true);
                clonedDiagram.style.maxWidth = '90%';
                clonedDiagram.style.maxHeight = '90%';
                clonedDiagram.style.cursor = 'zoom-out';

                // Add close functionality
                overlay.addEventListener('click', function() {
                    document.body.removeChild(overlay);
                });

                overlay.appendChild(clonedDiagram);
                document.body.appendChild(overlay);
            });
        });
    }, 1000);
});