const API_URL = 'http://localhost:3000/api';

async function testAuth() {
  const testUser = {
    email: `testuser_${Date.now()}@example.com`,
    password: 'Test123!',
    name: 'Test User'
  };

  console.log('Starting authentication test...');
  console.log('Test user:', testUser.email);

  try {
    // Test registration
    console.log('\n1. Testing registration...');
    const registerResponse = await fetch(`${API_URL}/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email: testUser.email,
        password: testUser.password,
        name: testUser.name,
      }),
    });

    const registerData = await registerResponse.json();
    
    if (!registerResponse.ok) {
      throw new Error(`Registration failed: ${JSON.stringify(registerData)}`);
    }
    
    console.log('✅ Registration successful:', registerData.message);
    
    // Test login with credentials
    console.log('\n2. Testing login with credentials...');
    const formData = new URLSearchParams();
    formData.append('email', testUser.email);
    formData.append('password', testUser.password);
    formData.append('redirect', 'false');
    formData.append('csrfToken', 'test-csrf-token'); // Required by NextAuth
    formData.append('json', 'true');
    
    const loginResponse = await fetch(`${API_URL}/auth/callback/credentials`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData.toString(),
    });

    const loginData = await loginResponse.json();
    
    if (!loginResponse.ok) {
      throw new Error(`Login failed: ${JSON.stringify(loginData)}`);
    }
    
    console.log('✅ Login successful:', loginData);
    
    // Test getting session
    console.log('\n3. Testing session...');
    const sessionResponse = await fetch(`${API_URL}/auth/session`);
    const sessionData = await sessionResponse.json();
    
    if (!sessionResponse.ok) {
      throw new Error(`Session check failed: ${JSON.stringify(sessionData)}`);
    }
    
    console.log('✅ Session data:', sessionData);
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    process.exit(1);
  }
}

testAuth();
