const fs = require('fs');
const content = fs.readFileSync('templates/index.html', 'utf8');
const start = content.indexOf('<script>');
const end = content.lastIndexOf('</script>');
const code = content.substring(start+8, end);
const lines = code.split('\n');

// Print last 50 lines with line numbers
const startLine = Math.max(0, lines.length - 50);
for (let i = startLine; i < lines.length; i++) {
  console.log(`${i+1}: ${lines[i]}`);
}
