/**
 * Update Widget Script
 * 
 * This script ensures the widget script is always up-to-date with the configuration.
 * It can be run as part of the build process to update the widget script.
 */

const fs = require('fs');
const path = require('path');

// Path to the widget script
const widgetScriptPath = path.join(__dirname, '../public/widget/chatbot-widget-new.js');

// Get the current date and time for versioning
const now = new Date();
const version = `${now.getFullYear()}.${now.getMonth() + 1}.${now.getDate()}-${now.getHours()}.${now.getMinutes()}`;

// Read the widget script
fs.readFile(widgetScriptPath, 'utf8', (err, data) => {
    if (err) {
        console.error('Error reading widget script:', err);
        process.exit(1);
    }

    // Update the version in the script
    const updatedData = data.replace(
        /const VERSION = ['"].*['"]/,
        `const VERSION = '${version}'`
    );

    // Write the updated script
    fs.writeFile(widgetScriptPath, updatedData, 'utf8', (err) => {
        if (err) {
            console.error('Error writing widget script:', err);
            process.exit(1);
        }

        console.log(`Widget script updated to version ${version}`);
    });
});

// Add the script to package.json scripts
// "update-widget": "node scripts/update-widget.js"
console.log('To add this script to your build process, add the following to your package.json scripts:');
console.log('"update-widget": "node scripts/update-widget.js"');
console.log('Then run it with: npm run update-widget'); 