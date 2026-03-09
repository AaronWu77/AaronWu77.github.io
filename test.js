const fs = require('fs');

const protectMath = (str, mathPlaceholder) => {
    str = str.replace(/\$\$([\s\S]*?)\$\$/g, function (match) {
        mathPlaceholder.push(match);
        return "%%%MATH_BLOCK_" + (mathPlaceholder.length - 1) + "%%%";
    });
    str = str.replace(/\$([^\n\$]+?)\$/g, function (match) {
        mathPlaceholder.push(match);
        return "%%%MATH_INLINE_" + (mathPlaceholder.length - 1) + "%%%";
    });
    return str;
};

const md = fs.readFileSync('CourseNotes/CourseNotes1/Markdown/CourseNotes1-1.md', 'utf-8');
const mathPlaceholder = [];
let protectedText = protectMath(md, mathPlaceholder);
console.log("Protect math done! Placesholders:", mathPlaceholder.length);
