const fs = require('fs');
const s = fs.readFileSync('templates/index.html','utf8');
const start = s.indexOf('<script>');
const end = s.lastIndexOf('</script>');
const code = s.substring(start+8,end);

function countToLine(lineNum) {
  const lines = code.split('\n');
  let scriptMod = lines.slice(0, lineNum).join('\n');
  
  const opening = {'{':0,'(':0,'[':0};
  const closing = {'}':0,')':0,']':0};
  
  for(let i=0; i<scriptMod.length; i++){
    const ch = scriptMod[i];
    if(ch==='"' || ch==="'"){
      const q = ch;
      i++;
      while(i<scriptMod.length){
        if(scriptMod[i]==='\\'){i+=2; continue;}
        if(scriptMod[i]===q) break;
        i++;
      }
      continue;
    }
    if(ch==='`'){
      i++;
      while(i<scriptMod.length){
        if(scriptMod[i]==='\\'){i+=2; continue;}
        if(scriptMod[i]==='`') break;
        i++;
      }
      continue;
    }
    if(opening.hasOwnProperty(ch)) opening[ch]++;
    if(closing.hasOwnProperty(ch)) closing[ch]++;
  }
  
  console.log(`Line ${lineNum}: {${opening['{']}-${closing['}']}} (${opening['{'] - closing['}']}) | (${opening['(']}-${closing[')']}) (${opening['('] - closing[')']}) | [${opening['[']}-${closing[']']}] (${opening['['] - closing[']']})`);
}

// Check at key lines
countToLine(657); // $.when
countToLine(976); // }).always
countToLine(1100); // middle
countToLine(1200); // near end
console.log('\nTotal lines:', code.split('\n').length);
