const fs = require('fs');
const s = fs.readFileSync('templates/index.html','utf8');
const start = s.indexOf('<script>');
const end = s.lastIndexOf('</script>');
const code = s.substring(start+8,end);

// Find the $.when line
const whenIdx = code.indexOf('$.when(');
const doneIdx = code.indexOf('.done(function(polygonsArgs, riskArgs)', whenIdx);
const alwaysIdx = code.indexOf('}).always(function', doneIdx);

console.log('$.when( at:', whenIdx);
console.log('.done( at:', doneIdx);
console.log('}).always( at:', alwaysIdx);
console.log('\nContext of .always():');
console.log(code.slice(alwaysIdx - 100, alwaysIdx + 150));
console.log('\n\nLines after .always():');
const afterAlways = code.slice(alwaysIdx + 200, alwaysIdx + 1000);
console.log(afterAlways.slice(0, 800));
