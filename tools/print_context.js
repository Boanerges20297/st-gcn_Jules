const fs = require('fs');
const s = fs.readFileSync('templates/index.html','utf8');
const start = s.indexOf('<script>');
const end = s.lastIndexOf('</script>');
const code = s.substring(start+8,end);
const positions = [17506,17540,27314,27335,27820,27821,28046,28112];
for(const p of positions){
  const from = Math.max(0,p-120);
  const to = Math.min(code.length,p+120);
  console.log('\n--- pos '+p+' ---\n');
  console.log(code.slice(from,to));
}
