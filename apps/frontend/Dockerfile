FROM node:20-alpine

WORKDIR /app

# Create non-root user for production hygiene
RUN addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001

COPY package*.json ./
RUN npm install

COPY . .

RUN npm run build

# Ensure nextjs user can read the application files
RUN chown -R nextjs:nodejs /app

USER nextjs

EXPOSE 3000

CMD ["npm", "start"]
