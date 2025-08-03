// Simple production server that bypasses TypeScript types
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');

// Import the built JavaScript file
const app = require('./dist/index').default;

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
