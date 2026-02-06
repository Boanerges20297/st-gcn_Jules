const fs = require('fs');
const s = fs.readFileSync('templates/index.html','utf8');
const start = s.indexOf('<script>');
const end = s.lastIndexOf('</script>');
const code = s.substring(start+8,end);
let lo = 0, hi = code.length;
let lastGood = 0;
while(lo < hi){
  const mid = Math.floor((lo+hi)/2);
  const chunk = code.slice(0, mid);
  try{
    new Function(chunk);
    lastGood = mid;
    lo = mid+1;
  }catch(e){
    hi = mid;
  }
}
console.log('lastGood', lastGood, 'nextChar', code.charCodeAt(lastGood), 'context:', code.slice(Math.max(0,lastGood-80), lastGood+80));
try{ new Function(code); console.log('full OK'); }catch(e){ console.log('full parse err', e.message); }
