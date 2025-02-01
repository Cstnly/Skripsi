const express = require('express');
const { createClient } = require('@supabase/supabase-js');
const bodyParser = require('body-parser');
require('dotenv').config();

const app = express();
app.use(bodyParser.json());

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_API_KEY;

// Initialize Supabase client
const supabase = createClient(supabaseUrl, supabaseKey);

async function UpdateDataPassword(username, encryptedPassword) {
    try {
        const { data, error } = await supabase
            .from('users')
            .update({ password_hashed: encryptedPassword })
            .eq('username', username);

        if (error) throw error;

        console.log(`Password updated successfully for user ${username}`);
        return data;
    } catch (err) {
        console.error('Failed to update password:', err);
        throw err;
    }
}

// Route to handle password update
app.post('/change_password', async (req, res) => {
    const { username, encryptedPassword } = req.body;

    try {
        if (!username || !encryptedPassword) {
            return res.status(400).json({ error: 'Username and encrypted password are required.' });
        }

        const data = await UpdateDataPassword(username, encryptedPassword);

        res.status(200).json({ message: 'Password updated successfully', data });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

const port = 5000;
app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
