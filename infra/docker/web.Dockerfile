# Dev-mode image for the Next.js dashboard. Production build/multistage image
# is a later-milestone concern (M1 only needs `next dev` behind Docker Compose).
FROM node:20-slim

WORKDIR /srv

COPY apps/web/package.json apps/web/package-lock.json* /srv/

RUN npm install

COPY apps/web /srv

EXPOSE 3000

CMD ["npm", "run", "dev"]
