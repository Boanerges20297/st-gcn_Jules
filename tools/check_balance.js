const fs = require('fs');
const s = fs.readFileSync('templates/index.html','utf8');
const start = s.indexOf('<script>');
const end = s.lastIndexOf('</script>');
if (start === -1 || end === -1) {
  console.error('script tags not found');
  process.exit(2);
}
const code = s.substring(start+8, end);
function check() {
  const stack = [];
  const pairs = { '{': '}', '[': ']', '(': ')' };
  for (let i = 0; i < code.length; i++) {
    const ch = code[i];
    if (ch === '\\' && i+1 < code.length) { i++; continue; }
    if (ch === '"' || ch === "'") { // skip strings
      const q = ch; i++;
      while (i < code.length) {
        if (code[i] === '\\') { i += 2; continue; }
        if (code[i] === q) { break; }
        i++; 
      }
      continue;
    }
    if (ch === '`') { // template literal
      i++;
      while (i < code.length) {
        if (code[i] === '\\') { i += 2; continue; }
        if (code[i] === '`') { break; }
        i++;
      }
      continue;
    }
    if (pairs[ch]) { stack.push({open: ch, pos: i}); }
    else if (ch === '}' || ch === ')' || ch === ']') {
      if (stack.length === 0) { console.log('Unmatched closing', ch, 'at', i); return; }
      const last = stack.pop();
      if (pairs[last.open] !== ch) { console.log('Mismatched', last.open, ch, 'at', i); return; }
    }
  }
  if (stack.length > 0) { console.log('Unclosed opens:', stack.map(x=>x.open+'@'+x.pos).join(', ')); }
  else console.log('All balanced');
}
check();
